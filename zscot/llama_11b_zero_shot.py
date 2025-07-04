import time
import json
import argparse
import re
import random
import requests
import numpy as np
import load_dotenv
import os

load_dotenv.load_dotenv()
api_key = os.getenv("TOGETHERAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the TOGETHERAI_API_KEY environment variable."


def extract_json(text):
    json_pattern = re.compile(r'{.*?}', re.DOTALL)

    match = json_pattern.search(text)

    if match:
        json_str = match.group()
        if json_str.strip():  # Check if the JSON string is not empty
            try:
                json_data = json.loads(json_str)
                return json_data
            except json.JSONDecodeError as e:
                return None

    return None


def validate_prediction(prediction: dict):
    class_labels = set({'Missed abnormality due to missing fixation', 'Missed abnormality due to reduced fixation',
                       'Missed abnormality due to incomplete knowledge', 'No missing abnormality'})

    valid_values = {0, 1}  # Set of valid values

    for label in class_labels:
        if label not in prediction:
            return False

        try:
            value = int(prediction[label])
        except (ValueError, TypeError):
            return False

        if value not in valid_values:
            return False

    return True


def model_request(system_content: str, prompt: str):
    url = "https://api.together.xyz/v1/chat/completions"

    payload = {
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    for _ in range(5):
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 503 or response.status_code == 500:
            time.sleep(1)
        else:
            return response.json()

    return None


def create_zero_shot_prompt(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps):
    prompt = f"""
    ### Experienced Radiologist:
    - Findings: {experienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(experienced_time_stamps, indent=2)}
    - Eye Gaze Data:
        - Fixations: {json.dumps(experienced_data['fixations'], indent=2)}
        - Durations: {json.dumps(experienced_data['durations'], indent=2)}

    ### Inexperienced Radiologist:
    - Findings: {inexperienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(inexperienced_time_stamps, indent=2)}
    - Eye Gaze Data:
        - Fixations: {json.dumps(inexperienced_data['fixations'], indent=2)}
        - Durations: {json.dumps(inexperienced_data['durations'], indent=2)}

    ### Task:
    Compare the findings, time-stamped text, and eye gaze data on the same medical imaging of both radiologists to address the following:

    1. Identify missed findings by the inexperienced radiologist.
    2. Analyze fixation and duration patterns to determine differences in focus and attention.

    Explain reasoning before returning the response.

    ### Response:
    Always provide an answer in JSON format, Answer: 1 for Yes and 0 for No:
    {{
        "Missed abnormality due to missing fixation": ,
        "Missed abnormality due to reduced fixation": ,
        "Missed abnormality due to incomplete knowledge": ,
        "No missing abnormality":
    }}
    """

    return prompt


def run_zshot_inference(data: dict, zshot_saved_predictions_to_JSON):
    system_role_prompt = "You are a helpful teaching assistant to provide feedback to the inexperienced radiologist."
    prediction_results = []
    skipped_dicom_predictions = set()

    inference_times = []
    number_processed = 0
    is_model_request_error = False

    for K in data.items():
        if K[0] in zshot_saved_predictions_to_JSON:
            continue

        pred_start_time = time.time()
        da = data[K[0]]['correct_data']

        result = [
            {
                'X_ORIGINAL': da['X_ORIGINAL'][i],
                'Y_ORIGINAL': da['Y_ORIGINAL'][i],
                'FPOGD': da['FPOGD'][i],
                'Time (in secs)': da['Time (in secs)'][i]
            }
            for i in range(len(da['X_ORIGINAL']))
        ]
        exp_fix = result.copy()
        s = ''
        exp_tran = da['transcript']
        for i in exp_tran:
            s = s + i['sentence']
        fixations_dur = da['FPOGD']

        fixations_exp = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        da = data[K[0]]['incorrect_data']
        if len(da) == 0:
            da = data[K[0]]['correct_data']
            si = ''
            inexp_tran = da['transcript']
            for i in inexp_tran:
                si = si + i['sentence']
            fake_durations = da['FPOGD']
            fake_fixations = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        else:
            si = ''
            inexp_tran = da['transcript']
            for i in inexp_tran:
                si = si + i['sentence']
            fake_durations = da['FPOGD']
            fake_fixations = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        # Prepare the data
        experienced_data = {
            'transcript': s,
            'fixations': fixations_exp,
            'durations': fixations_dur
        }

        inexperienced_data = {
            'transcript': si,
            'fixations': fake_fixations,
            'durations': fake_durations
        }

        prompt = create_zero_shot_prompt(
            experienced_data, inexperienced_data, exp_tran, inexp_tran)

        response = model_request(system_role_prompt, prompt)

        if response is None:
            # print("Error in response.")
            skipped_dicom_predictions.add(K[0])
            continue

        # print(response)

        if 'error' in response:
            print("There is an error in with TogetherAI response:",
                  response['error'])

            is_model_request_error = True
            return prediction_results, skipped_dicom_predictions, inference_times, is_model_request_error

        error_assessment = extract_json(['choices'][0]['message']['content'])

        if error_assessment is None:
            # print("Error in extracting JSON from response.")
            skipped_dicom_predictions.add(K[0])
            continue
        else:
            if not validate_prediction(error_assessment):
                # print("Invalid prediciton reponse JSON missing label.")
                skipped_dicom_predictions.add(K[0])
            else:
                # print(K[0], error_assessment)
                zshot_saved_predictions_to_JSON[K[0]] = error_assessment
                prediction_results.append(error_assessment)
                pred_end_time = time.time()
                inference_times.append(pred_end_time - pred_start_time)
                number_processed += 1
                print("Number Processed:", number_processed)

    return prediction_results, skipped_dicom_predictions, inference_times, is_model_request_error


def main():
    parser = argparse.ArgumentParser(
        description="Your script description here.")
    parser.add_argument('--metadata', type=str,
                        help='Path to dataset transcript metadata file', required=True)
    parser.add_argument('--data', type=str,
                        help='Path to transcript data file', required=True)
    parser.add_argument('--results', type=str,
                        help='Path to preexisting results file', required=False)
    args = parser.parse_args()

    with open(args.metadata, 'r') as file:
        datalab = json.load(file)

    with open(args.data, 'r') as file:
        data = json.load(file)

    assert len(datalab) == len(
        data), "Metadata and data lengths do not match"

    random_sampled_data = {key: data[key]
                           for key in random.sample(list(data.keys()), len(data))}
    assert len(random_sampled_data) == len(data)
    dataset_size = len(random_sampled_data)

    batch_size = int(dataset_size / 20)
    batches = [{} for _ in range(20)]

    for i, (key, value) in enumerate(random_sampled_data.items()):
        batch_index = i // batch_size
        if batch_index >= 20:
            batch_index = 19
        batches[batch_index][key] = value

    print("Batch Size", batch_size, "| Last Batch Size", len(batches[-1]))
    print("Total Number of Samples", sum(len(batch) for batch in batches))

    assert sum(len(batch) for batch in batches) == dataset_size

    # Run the zero-shot inference and save the results
    zshot_saved_predictions_to_JSON = {}
    results_output_file = args.results if args.results else "llama_11B_vision_zshot_results.json"

    if args.results:
        with open(args.results, 'r') as file:
            zshot_saved_predictions_to_JSON = json.load(file)

    sample_runtimes = []
    current_batch = 1
    is_model_request_error = False

    for b in batches:
        print("Inference on batch", current_batch)

        zshot_results, zshot_skipped_dicom_ids, inference_times, has_error = run_zshot_inference(
            b, zshot_saved_predictions_to_JSON)

        if has_error:
            is_model_request_error = True
            break

        sample_runtimes.extend(inference_times)
        print("Results", len(zshot_results),
              "| Skipped", len(zshot_skipped_dicom_ids))
        print("--------------------------------------------")
        current_batch += 1

    with open(results_output_file, 'w') as file:
        json.dump(zshot_saved_predictions_to_JSON, file, indent=4)

    np.save("sample_runtimes.npy", np.array(sample_runtimes))

    if is_model_request_error:
        print(
            "There is an error with TogetherAI request no results returned from the model.")
        return

    # Perform a final check to see if all data has been processed
    missed_samples = {}

    for key in random_sampled_data.keys():
        if key not in zshot_saved_predictions_to_JSON.keys():
            missed_samples[key] = random_sampled_data[key]

    print("Missed samples:", len(missed_samples))

    missed_batches = []

    for i in range(0, len(missed_samples), batch_size):
        missed_batches.append({k: missed_samples[k] for k in list(
            missed_samples.keys())[i:i + batch_size]})

    epochs = 0

    while len(zshot_saved_predictions_to_JSON) < len(random_sampled_data) and epochs < 5 and not is_model_request_error:
        for b in missed_batches:
            zshot_results, zshot_skipped_dicom_ids, inference_times, has_error = run_zshot_inference(
                b, zshot_saved_predictions_to_JSON)
            sample_runtimes.extend(inference_times)

            if has_error:
                is_model_request_error = True
                break

        epochs += 1

    with open(results_output_file, 'w') as file:
        json.dump(zshot_saved_predictions_to_JSON, file, indent=4)

    np.save("sample_runtimes.npy", np.array(sample_runtimes))

    if is_model_request_error:
        print(
            "There is an error with TogetherAI request no results returned from the model.")
        return

    print("DONE")


if __name__ == "__main__":
    main()
