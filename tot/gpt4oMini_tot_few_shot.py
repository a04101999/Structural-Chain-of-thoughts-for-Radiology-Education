import time
import json
import argparse
import random
import load_dotenv
import os
from model_request.req import extract_json, validate_prediction, openai_request

load_dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the OPENAI_API_KEY environment variable."


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


def create_tot_prompt_few_shot(experienced_data: dict, inexperienced_data: dict, experienced_time_stamps: dict, inexperienced_time_stamps: dict, ex_experienced_data: dict, ex_inexperienced_data: dict, ex_experienced_time_stamps: dict, ex_inexperienced_time_stamps: dict, ex_correct_output: dict):
    prompt = f"""
    You are an expert radiology educator AI comparing findings and eye gaze patterns of two radiologists to identify missed abnormalities.
    
    ## Provided is an example of an experienced and inexperienced radiologist's findings, time-stamped text, and eye gaze data on a medical imaging.

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

    ### Answer: The inexperienced radiologist missed the following abnormalities due to these reasons.
    - {json.dumps(ex_correct_output, indent=4)}

    ---

    ## Now, compare the findings, time-stamped text, and eye gaze data of the two radiologists on the same medical imaging.
    
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
    Compare the findings, time-stamped text, and eye gaze data on the same medical imaging of both radiologists to address the following provided below:
    
    1. Identify missed findings by the inexperienced radiologist.
    2. Analyze fixation and duration patterns to determine differences in focus and attention.

    Use **Tree of Thoughts-style reasoning** to identify and explain the cause of missed findings by the inexperienced radiologist:
    
    #### Step 1: Generate 4 possible hypotheses (thoughts) for why a finding may have been missed or if there is no error. For example:
    - A: Missed due to no fixation.
    - B: Missed due to brief fixation.
    - C: Fixated appropriately, but not reported due to incomplete knowledge.
    - D: There is no error.
    
    #### Step 2: For each hypothesis, analyze whether the data supports it based on gaze fixations, durations, and time-stamped text.
    
    #### Step 3: Choose the most likely explanation(s) and explain why it is more plausible than others.
    
    ---
    
    ### Output:
    Return your answer as a structured JSON object. Use 1 for Yes and 0 for No.
    {{
        "Missed abnormality due to missing fixation": ,
        "Missed abnormality due to reduced fixation": ,
        "Missed abnormality due to incomplete knowledge": ,
        "No missing abnormality": 
    }}
    """
    return prompt


def run_tot_inference(current_batch: dict, all_data: dict, metadata: dict, ex_dicom_id: str, saved_predictions_to_JSON: dict):
    system_role_prompt = "You are a helpful teaching assistant to provide feedback to the inexperienced radiologist."
    number_processed = 0

    class_labels = set({'Missed abnormality due to missing fixation', 'Missed abnormality due to reduced fixation',
                       'Missed abnormality due to incomplete knowledge', 'No missing abnormality'})

    for K in current_batch.items():
        if K[0] in saved_predictions_to_JSON:
            continue

        pred_start_time = time.time()
        da = current_batch[K[0]]['correct_data']

        s = ''
        exp_tran = da['transcript']
        for i in exp_tran:
            s = s + i['sentence']
        fixations_dur = da['FPOGD']

        fixations_exp = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        da = current_batch[K[0]]['incorrect_data']
        if len(da) == 0:
            da = current_batch[K[0]]['correct_data']
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
            all_data, ex_dicom_id)
        ex_correct_output = get_correct_class_labels(metadata[ex_dicom_id])

        prompt = create_tot_prompt_few_shot(
            experienced_data, inexperienced_data, exp_tran, inexp_tran, ex_experienced_data, ex_inexperienced_data, ex_experienced_time_stamps, ex_inexperienced_time_stamps, ex_correct_output)

        response = openai_request(system_role_prompt, prompt, api_key,
                                  model="gpt-4o-mini", temperature=0.2, max_tokens=500)

        if response is None:
            # print("Error in response.")
            continue

        # print(response)

        error_assessment = extract_json(response)

        if error_assessment is None:
            # print("Error in extracting JSON from response.")
            continue

        if validate_prediction(error_assessment, class_labels):
            pred_end_time = time.time()
            saved_predictions_to_JSON[K[0]] = error_assessment
            saved_predictions_to_JSON[K[0]
                                      ]['inference_time'] = pred_end_time - pred_start_time
            print(K[0], "processed in", pred_end_time -
                  pred_start_time, "seconds.")
            number_processed += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GPT-4o-Mini few shot tree of thoughts inference on dataset.")
    parser.add_argument('--metadata', type=str,
                        help='Path to dataset transcript metadata file', required=True)
    parser.add_argument('--data', type=str,
                        help='Path to transcript data file', required=True)
    parser.add_argument('--results', type=str,
                        help='Path to preexisting results file', required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.metadata, 'r') as file:
        datalab = json.load(file)
    with open(args.data, 'r') as file:
        data = json.load(file)

    random_sampled_data = {key: data[key]
                           for key in random.sample(list(data.keys()), len(data))}

    dataset_size = len(random_sampled_data)
    batch_size = int(dataset_size / 20)
    batches = [{} for _ in range(20)]

    random_id = random.randint(0, dataset_size - 1)
    ex_dicom_id = list(random_sampled_data.keys())[random_id]

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
    tot_saved_predictions_to_JSON = {}
    results_output_file = args.results if args.results else "gpt4o_mini_tot_fscot_results.json"

    if args.results:
        print("Loading existing results from", args.results)

        with open(args.results, 'r') as file:
            tot_saved_predictions_to_JSON = json.load(file)

        print("Loaded", len(tot_saved_predictions_to_JSON),
              "existing predictions.")
        print("===============================================\n")

    current_batch = 1

    for b in batches:
        print("Inference on batch", current_batch)
        run_tot_inference(
            b, data, datalab, ex_dicom_id, tot_saved_predictions_to_JSON)

        print("Completed batch", current_batch)
        current_batch += 1
        with open(results_output_file, 'w') as file:
            json.dump(tot_saved_predictions_to_JSON, file, indent=4)

        print("===============================================\n")

    with open(results_output_file, 'w') as file:
        json.dump(tot_saved_predictions_to_JSON, file, indent=4)

    # Perform a final check to see if all data has been processed
    missed_samples = {}

    for key in random_sampled_data.keys():
        if key not in tot_saved_predictions_to_JSON.keys():
            missed_samples[key] = random_sampled_data[key]

    print("Number of missed samples:", len(missed_samples))
    print("===============================================\n")

    missed_batches = []

    for i in range(0, len(missed_samples), batch_size):
        missed_batches.append({k: missed_samples[k] for k in list(
            missed_samples.keys())[i:i + batch_size]})

    epochs = 0

    while len(tot_saved_predictions_to_JSON) < len(random_sampled_data) and epochs < 5:
        for i, b in enumerate(missed_batches):
            run_tot_inference(
                b, data, datalab, ex_dicom_id, tot_saved_predictions_to_JSON)

            print(f"Completed batch {i + 1}")
            with open(results_output_file, 'w') as file:
                json.dump(tot_saved_predictions_to_JSON, file, indent=4)
            print("===============================================\n")

        epochs += 1

    with open(results_output_file, 'w') as file:
        json.dump(tot_saved_predictions_to_JSON, file, indent=4)

    print("DONE")


if __name__ == "__main__":
    main()
