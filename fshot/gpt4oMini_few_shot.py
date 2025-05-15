import time
import json
import argparse
import re
from openai import OpenAI
import os
import random


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
    # Place your OpenAI API key here
    api_key = ""
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return completion.choices[0].message.content


def create_few_shot_prompt(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps, ex_experienced_data, ex_inexperienced_data, ex_experienced_time_stamps, ex_inexperienced_time_stamps, ex_correct_output):
    prompt = f"""
    Provided is an example of an experienced and inexperienced radiologist's findings, time-stamped text, and eye gaze data on a medical imaging.

    ### Experienced Radiologist:
    - Findings: {ex_experienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(ex_experienced_time_stamps, indent=4)}
    - Eye Gaze Data: 
        - Fixations: {json.dumps(ex_experienced_data['fixations'], indent=4)}
        - Durations: {json.dumps(ex_experienced_data['durations'], indent=4)}

    ### Inexperienced Radiologist:
    - Findings: {ex_inexperienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(ex_inexperienced_time_stamps, indent=4)}
    - Eye Gaze Data: 
        - Fixations: {json.dumps(ex_inexperienced_data['fixations'], indent=4)}
        - Durations: {json.dumps(ex_inexperienced_data['durations'], indent=4)}

    ### Answer: The inexperienced radiologist missed the following abnormalities due to these reasons
    - {json.dumps(ex_correct_output, indent=4)}


    ### Task:
    Compare the findings, time-stamped text, and eye gaze data on the same medical imaging of both radiologists  to address the following provided below:
    
    1. Identify missed findings by the inexperienced radiologist.
    2. Analyze fixation and duration patterns to determine differences in focus and attention.

    ### Experienced Radiologist:
    - Findings: {experienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(experienced_time_stamps, indent=4)}
    - Eye Gaze Data: 
        - Fixations: {json.dumps(experienced_data['fixations'], indent=4)}
        - Durations: {json.dumps(experienced_data['durations'], indent=4)}

    ### Inexperienced Radiologist:
    - Findings: {inexperienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(inexperienced_time_stamps, indent=4)}
    - Eye Gaze Data: 
        - Fixations: {json.dumps(inexperienced_data['fixations'], indent=4)}
        - Durations: {json.dumps(inexperienced_data['durations'], indent=4)}
    
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


def transform_data_for_prompt(data, dicom_id):
    da = data[dicom_id]['correct_data']
    s = ''
    exp_tran = da['transcript']
    for i in exp_tran:
        s = s + i['sentence']

    fixations_dur = da['FPOGD']

    fixations_exp = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

    da = data[dicom_id]['incorrect_data']
    if len(da) == 0:
        da = data[dicom_id]['correct_data']
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

    return experienced_data, inexperienced_data, exp_tran, inexp_tran


def get_correct_class_labels(fixation_metadata):
    return {
        "Missed abnormality due to missing fixation": fixation_metadata['class_label_1'],
        "Missed abnormality due to reduced fixation": fixation_metadata['class_label_2'],
        "Missed abnormality due to incomplete knowledge": fixation_metadata['class_label_3'],
        "No missing abnormality": 1 if (fixation_metadata['class_label_1'] == 0 and fixation_metadata['class_label_2'] == 0 and fixation_metadata['class_label_3'] == 0) else 0
    }


def run_fshot_inference(data: dict, agg_data, datalab, ex_dicom_id: str, fshot_saved_predictions_to_JSON):
    system_role_prompt = "You are a helpful teaching assistant to provide feedback to the inexperienced radiologist."
    prediction_results = []
    skipped_dicom_predictions = set()

    inference_times = []
    number_processed = 0

    for K in data.items():
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

        ex_experienced_data, ex_inexperienced_data, ex_experienced_time_stamps, ex_inexperienced_time_stamps = transform_data_for_prompt(
            agg_data, ex_dicom_id)
        ex_correct_output = get_correct_class_labels(datalab[ex_dicom_id])

        prompt = create_few_shot_prompt(
            experienced_data, inexperienced_data, exp_tran, inexp_tran, ex_experienced_data, ex_inexperienced_data, ex_experienced_time_stamps, ex_inexperienced_time_stamps, ex_correct_output)

        response = model_request(system_role_prompt, prompt)

        if response is None:
            skipped_dicom_predictions.add(K[0])
            continue

        # print(response)

        error_assessment = extract_json(response)

        if error_assessment is None:
            skipped_dicom_predictions.add(K[0])
            continue
        else:
            if not validate_prediction(error_assessment):
                skipped_dicom_predictions.add(K[0])
            else:
                fshot_saved_predictions_to_JSON[K[0]] = error_assessment
                prediction_results.append(error_assessment)
                pred_end_time = time.time()
                inference_times.append(pred_end_time - pred_start_time)
                number_processed += 1

    return prediction_results, skipped_dicom_predictions, inference_times


def main():
    parser = argparse.ArgumentParser(
        description="Your script description here.")
    parser.add_argument('--metadata', type=str,
                        help='Path to dataset transcript metadata file', required=True)
    parser.add_argument('--data', type=str,
                        help='Path to transcript data file', required=True)
    args = parser.parse_args()

    print(f"Metadata file: {args.metadata}")
    print(f"Data file: {args.data}")

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

    random_number = random.randint(0, dataset_size - 1)
    ex_dicom_id = list(random_sampled_data.keys())[random_number]

    for i, (key, value) in enumerate(random_sampled_data.items()):
        batch_index = i // batch_size
        if batch_index >= 20:
            batch_index = 19
        batches[batch_index][key] = value

    print("Batch Size", batch_size, "| Last Batch Size", len(batches[-1]))
    print("Total Number of Samples", sum(len(batch) for batch in batches))

    assert sum(len(batch) for batch in batches) == dataset_size

    # Run the zero-shot inference and save the results

    fshot_saved_predictions_to_JSON = {}
    sample_runtimes = []
    current_batch = 1

    for b in batches:
        print("Inference on batch", current_batch)
        scot_results, scot_skipped_dicom_ids, inference_times = run_fshot_inference(
            b, data, datalab, ex_dicom_id, fshot_saved_predictions_to_JSON)
        sample_runtimes.extend(inference_times)
        print("Results", len(scot_results),
              "| Skipped", len(scot_skipped_dicom_ids))
        print("--------------------------------------------")
        current_batch += 1

    with open("gpt4o_mini_fshot_results.json", 'w') as file:
        json.dump(fshot_saved_predictions_to_JSON, file, indent=4)

    # Perform a final check to see if all data has been processed
    missed_samples = {}

    for key in random_sampled_data.keys():
        if key not in fshot_saved_predictions_to_JSON.keys():
            missed_samples[key] = random_sampled_data[key]

    print("Missed samples:", len(missed_samples))

    missed_batches = []

    for i in range(0, len(missed_samples), batch_size):
        missed_batches.append({k: missed_samples[k] for k in list(
            missed_samples.keys())[i:i + batch_size]})

    epochs = 0

    while len(fshot_saved_predictions_to_JSON) < len(random_sampled_data) and epochs < 5:
        for b in missed_batches:
            scot_results, scot_skipped_dicom_ids, inference_times = run_fshot_inference(
                b, fshot_saved_predictions_to_JSON)
            sample_runtimes.extend(inference_times)

        epochs += 1

    with open("gpt4o_mini_fshot_results.json", 'w') as file:
        json.dump(fshot_saved_predictions_to_JSON, file, indent=4)

    print("DONE | Average Inference runtime", (round(
        sum(sample_runtimes) / len(sample_runtimes), 3)), "seconds per sample")


if __name__ == "__main__":
    main()
