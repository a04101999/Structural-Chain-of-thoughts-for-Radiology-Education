import time
import json
import argparse
import re
from openai import OpenAI
import random
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the OPENAI_API_KEY environment variable."


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


def create_got_prompt(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps):
    prompt = f"""
    You are an expert AI assistant analyzing radiologist behavior using eye gaze and transcript data. Your task is to identify whether the inexperienced radiologist missed any abnormalities and why — using a **Graph of Thoughts** reasoning strategy.
    
    ---
    
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
    
    ---
    
    ### Task:
    Use a **Graph of Thoughts** reasoning process:
    
    #### Step 1: Construct nodes for each possible explanation:
    - A: Missed due to no fixation.
    - B: Missed due to brief fixation.
    - C: Fixated appropriately, but not reported due to incomplete knowledge.
    - D: Finding is not abnormal / no error.
    
    #### Step 2: For each node, extract supporting or contradicting evidence (fixation duration, timestamps, verbal mention).
    
    #### Step 3: Construct a reasoning graph by linking nodes that:
    - Support each other (A → B if one leads to another)
    - Contradict each other (A ⟷ ¬D)
    - Share evidence (A ⟷ C if both depend on fixation absence)
    
    #### Step 4: Analyze this graph to decide which explanation(s) are most supported.
    
    ---
    
    ### Output:
    Return your answer as a structured JSON object. Use 1 for Yes and 0 for No:
    {{
        "Missed abnormality due to missing fixation": ,
        "Missed abnormality due to reduced fixation": ,
        "Missed abnormality due to incomplete knowledge": ,
        "No missing abnormality":
    }}
    """

    return prompt


def run_got_inference(data: dict, got_saved_predictions_to_JSON):
    system_role_prompt = "You are a helpful teaching assistant to provide feedback to the inexperienced radiologist."
    number_processed = 0

    for K in data.items():
        if K[0] in got_saved_predictions_to_JSON:
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

        prompt = create_got_prompt(
            experienced_data, inexperienced_data, exp_tran, inexp_tran)

        response = model_request(system_role_prompt, prompt)

        if response is None:
            # print("Error in response.")
            continue

        # print(response)

        error_assessment = extract_json(response)

        if error_assessment is None:
            # print("Error in extracting JSON from response.")
            continue

        if validate_prediction(error_assessment):
            # print(K[0], error_assessment)
            pred_end_time = time.time()
            got_saved_predictions_to_JSON[K[0]] = error_assessment
            got_saved_predictions_to_JSON[K[0]
                                          ]['inference_time'] = pred_end_time - pred_start_time
            print(K[0], "processed in", pred_end_time -
                  pred_start_time, "seconds.")
            number_processed += 1


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
    print("===============================================\n")

    assert sum(len(batch) for batch in batches) == dataset_size

    # Run the zero-shot inference and save the results
    got_saved_predictions_to_JSON = {}
    results_output_file = args.results if args.results else "gpt4o_mini_got_results.json"

    if args.results:
        print("Loading existing results from", args.results)
        with open(args.results, 'r') as file:
            got_saved_predictions_to_JSON = json.load(file)

        print("Loaded", len(got_saved_predictions_to_JSON),
              "existing predictions.")
        print("===============================================\n")

    current_batch = 1

    for b in batches:
        print("Inference on batch", current_batch)
        run_got_inference(
            b, got_saved_predictions_to_JSON)

        print("Completed batch", current_batch)
        with open(results_output_file, 'w') as file:
            json.dump(got_saved_predictions_to_JSON, file, indent=4)
        current_batch += 1

        print("===============================================\n")

    with open(results_output_file, 'w') as file:
        json.dump(got_saved_predictions_to_JSON, file, indent=4)

    # Perform a final check to see if all data has been processed
    missed_samples = {}

    for key in random_sampled_data.keys():
        if key not in got_saved_predictions_to_JSON.keys():
            missed_samples[key] = random_sampled_data[key]

    print("Missed samples:", len(missed_samples))
    print("===============================================\n")

    missed_batches = []

    for i in range(0, len(missed_samples), batch_size):
        missed_batches.append({k: missed_samples[k] for k in list(
            missed_samples.keys())[i:i + batch_size]})

    epochs = 0

    while len(got_saved_predictions_to_JSON) < len(random_sampled_data) and epochs < 5:
        for i, b in enumerate(missed_batches):
            run_got_inference(
                b, got_saved_predictions_to_JSON)

            print(f"Completed batch {i + 1}")
            with open(results_output_file, 'w') as file:
                json.dump(got_saved_predictions_to_JSON, file, indent=4)
            print("===============================================\n")

        epochs += 1

    with open(results_output_file, 'w') as file:
        json.dump(got_saved_predictions_to_JSON, file, indent=4)

    print("DONE")


if __name__ == "__main__":
    main()
