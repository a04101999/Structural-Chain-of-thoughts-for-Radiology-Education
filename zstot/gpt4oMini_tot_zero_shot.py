import time
import json
import argparse
import re
from openai import OpenAI
import random
from dotenv import load_dotenv
import os
from model_request.req import openai_request, extract_json, validate_prediction

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the OPENAI_API_KEY environment variable."

TREE_DEPTH = 2                # number of steps
PROPOSAL_RUNS_PER_STATE = 2   # number of new hypotheses per state
EVAL_RUNS_PER_STATE = 2       # number of times to evaluate each hypothesis
BRANCH_FACTOR = 3             # keep top-k at each level
class_labels = set({'Missed abnormality due to missing fixation', 'Missed abnormality due to reduced fixation',
                    'Missed abnormality due to incomplete knowledge', 'No missing abnormality'})


def create_tot_prompt(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps):
    prompt = f"""
    You are an expert radiology educator AI comparing findings and eye gaze patterns of two radiologists to identify missed abnormalities.

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
    Compare the findings, time-stamped text, and eye gaze data on the same medical imaging of both radiologists to address:

    1. Identify missed findings by the inexperienced radiologist.
    2. Analyze fixation and duration patterns to determine differences in focus and attention.

    Use **Tree of Thoughts-style reasoning**:
    #### Step 1: Generate 4 possible hypotheses:
    - A: Missed due to no fixation.
    - B: Missed due to brief fixation.
    - C: Fixated appropriately, but not reported due to incomplete knowledge.
    - D: There is no error.

    #### Step 2: For each hypothesis, analyze whether the data supports it.
    #### Step 3: Choose the most likely explanation(s) and explain why.

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


def run_llm(prompt):
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert radiology educator AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return completion.choices[0].message.content


def generate_thoughts(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps):
    prompt = create_tot_prompt(
        experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps
    )
    answer = run_llm(prompt)
    return answer.strip()


def evaluate_thought(hypothesis_text):
    """
    Ask LLM: how plausible is this explanation? Output score between 0 and 1.
    """
    prompt = f"""
    Evaluate the plausibility (0=impossible, 1=very likely) of this hypothesis about why a radiologist missed an abnormality:

    "{hypothesis_text}"

    Output only the score as a decimal between 0 and 1.
    """
    answer = run_llm(prompt)
    try:
        return float(answer.strip())
    except Exception:
        return 0.0


def tree_of_thoughts_analysis(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps):
    curr_thoughts = [""]  # initial state
    analysis_start_time = time.time()

    for _ in range(TREE_DEPTH):
        proposal_and_score = []

        for _ in curr_thoughts:
            # generate proposals
            proposals = []

            while len(proposals) < PROPOSAL_RUNS_PER_STATE:
                proposal = generate_thoughts(
                    experienced_data, inexperienced_data,
                    experienced_time_stamps, inexperienced_time_stamps
                )
                error_assessment = extract_json(proposal)

                if error_assessment is None or not validate_prediction(error_assessment, class_labels):
                    continue

                proposals.append(proposal)

            # evaluate each
            for proposal in proposals:
                scores = [
                    evaluate_thought(proposal) for _ in range(EVAL_RUNS_PER_STATE)
                ]
                avg_score = sum(scores) / len(scores)
                proposal_and_score.append((proposal, avg_score))

        # sort and keep top‑k
        proposal_and_score.sort(key=lambda x: x[1], reverse=True)
        curr_thoughts = [p for p, _ in proposal_and_score[:BRANCH_FACTOR]]

    analysis_end_time = time.time()
    return curr_thoughts[0], analysis_end_time - analysis_start_time


def run_inference(data: dict, saved_predictions_to_JSON: dict):
    number_processed = 0

    for K in data.items():
        if K[0] in saved_predictions_to_JSON:
            continue

        pred_start_time = time.time()
        da = data[K[0]]['correct_data']

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

        response, tot_analysis_time = tree_of_thoughts_analysis(
            experienced_data, inexperienced_data, exp_tran, inexp_tran)

        if response is None:
            continue

        # print(response)

        error_assessment = extract_json(response)

        if error_assessment is None:
            continue

        if validate_prediction(error_assessment, class_labels):
            pred_end_time = time.time()
            saved_predictions_to_JSON[K[0]] = error_assessment
            saved_predictions_to_JSON[K[0]
                                      ]['inference_time'] = pred_end_time - pred_start_time
            saved_predictions_to_JSON[K[0]
                                      ]['tot_analysis_time'] = tot_analysis_time

            print(K[0], "processed in", pred_end_time -
                  pred_start_time, "seconds.")
            number_processed += 1


def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-4o-Mini inference on TOT dataset.")
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
    saved_predictions_to_JSON = {}
    results_output_file = args.results if args.results else "gpt4o_mini_tot_zero_shot_results.json"

    if args.results:
        print("Loading existing results from", args.results)

        with open(args.results, 'r') as file:
            saved_predictions_to_JSON = json.load(file)

        print("Loaded", len(saved_predictions_to_JSON),
              "existing predictions.")
        print("===============================================\n")

    current_batch = 1

    for b in batches:
        print("Inference on batch", current_batch)
        run_inference(
            b, saved_predictions_to_JSON)

        print("Completed batch", current_batch)
        current_batch += 1
        with open(results_output_file, 'w') as file:
            json.dump(saved_predictions_to_JSON, file, indent=4)

        print("===============================================\n")

    with open(results_output_file, 'w') as file:
        json.dump(saved_predictions_to_JSON, file, indent=4)

    # Perform a final check to see if all data has been processed
    missed_samples = {}

    for key in random_sampled_data.keys():
        if key not in saved_predictions_to_JSON.keys():
            missed_samples[key] = random_sampled_data[key]

    print("Number of missed samples:", len(missed_samples))
    print("===============================================\n")

    missed_batches = []

    for i in range(0, len(missed_samples), batch_size):
        missed_batches.append({k: missed_samples[k] for k in list(
            missed_samples.keys())[i:i + batch_size]})

    epochs = 0

    while len(saved_predictions_to_JSON) < len(random_sampled_data) and epochs < 5:
        for i, b in enumerate(missed_batches):
            run_inference(
                b, saved_predictions_to_JSON)

            print(f"Completed batch {i + 1}")
            with open(results_output_file, 'w') as file:
                json.dump(saved_predictions_to_JSON, file, indent=4)
            print("===============================================\n")

        epochs += 1

    with open(results_output_file, 'w') as file:
        json.dump(saved_predictions_to_JSON, file, indent=4)

    print("DONE")


if __name__ == "__main__":
    main()
