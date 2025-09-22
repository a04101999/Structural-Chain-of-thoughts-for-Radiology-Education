[This repo is still under developement]

# Structural-Chain-of-thoughts-for-Radiology-Education

Radiology education requires trainees to develop both perceptual and interpretive expertise, yet the feedback required to develop these skills remain scarce due to the demanding schedules of experienced radiologists. This lack of personalized guidance makes it difficult for learners to understand
not just what errors they made, but also the reason why those errors occurred and how to refine their reasoning skills. Although Large Language Models (LLMs) and Large Multimodal Models (LMMs) have shown promise in radiology applications, they struggle with fine-grained multimodal reasoning. Specifically, these models struggle in detecting subtle cross-modal patterns, such as variations in gaze behavior and diagnostic decisions. These small yet critical differences in how experts and novices allocate visual attention can reveal underlying perceptual gaps, which are often overlooked by current AI-driven approaches. To address these limitations, we introduce Structural Chain of Thoughts (SCoT), a novel framework that enhances AI sensitivity to nuanced multimodal differences by structuring gaze data
and diagnostic reasoning into a thought graph. By leveraging a structural prior, SCoT systematically identifies key perceptual and interpretive discrepancies, allowing models to provide targeted, context-aware feedback. This structured approach not only highlights missed findings but also explains
the reasoning behind perceptual errors, turning them into learning opportunities. Applied within radiology education, SCoT bridges the gap between expert and novice performance, offering a scalable solution for AI-driven diagnostic training. We further contribute a simulated dataset of perceptual
errors, facilitating future research into multimodal reasoning and educational
AI in medical imaging.

## Table of Contents

- [Dataset](#dataset)
- [SCoT Framework](#framework)
- [Usage](#usage)

# Simulated Error Dataset: <a name="dataset"></a>

Due to unavailability of Real world error dataset, this study was conducted on the simulated error dataset:

https://drive.google.com/drive/folders/1RzlGzvJ9Dl01dgrNlhNedY7JWhQ3pmJE?usp=sharing

It contains two files:

1. Error dataset with missing or masked fixations
2. Labels for the the cases

## Dataset Generation

This dataset was derived from the [EGD-CXR](https://physionet.org/content/egd-cxr/1.0.0/) dataset. The EGD-CXR dataset was created using an eye-tracking system to monitor a radiologist's gaze while interpreting and reading 1,083 publicly available chest X-ray (CXR) images. We provide a Python file within the `dataset_generation` directory with the code used to generate the simulated error dataset. To reproduce the synthesized error dataset, you will need to download the original EGD-CXR dataset from PhysioNet and extract the audio_segmentation_transcripts and fixation folders as displayed within the `dataset_generation` directory. Then run the `egd_cxr_processing.py` file with the following command:

```bash
python3 egd_cxr_processing.py
```

For convenience we have provided the respective folders in the `dataset_generation` directory and simply running the `egd_cxr_processing.py` file will generate the synthesized error dataset.

# Implementation of the SCoT Framework: <a name="framework"></a>

The Python implementation of the SCoT (Structural Chain of Thoughts) framework used to
identify nuanced multimodal differences by structuring gaze data and diagnostic reasoning
into a thought graph is located in this [Python file](https://github.com/a04101999/Structural-Chain-of-thoughts-for-Radiology-Education/blob/main/scot/scot_framework/scot_creation.py).

# Usage <a name="usage"></a>

Within this repository, we provide python files that utilizes the synthesized perceptual error dataset and evaluates the SCoT framework against standard chain of thought (CoT) prompting in zero-shot and few-shot settings on the synthesized error dataset highlighting its effectiveness in improving multimodal reasoning across different LLM/LMM models. In this study the models used are Mistral-7B-Instruct-v0.3, LLAMA-3.2-11B-Vision-Instruct and GPT-4o-Mini. The Mistral and Llama models are accessed using [together.ai's](https://www.together.ai/) API. GPT-4o-Mini is accessed using [OpenAI's](https://openai.com/api/) API.

- These python files are located in the `./zscot` (Zero Shot Chain of Thoughts), `./fscot` (Few Shot Chain of Thoughts), `./scot` (Structural Chain of Thoughts) and `./zstot` (Zero Shot Tree of Thoughts) directories of this repository.
- To run these programs you will need a together.ai API key and OpenAI API key to perform requests to these models.
- These API keys can be obtained by signing up for an account on the respective platforms and following their instructions for generating API keys.
- These experiments were performed using Python 3.8 or higher.

## Set Up .env File

To run the code, you will need to create a `.env` file in the root directory of the repository. This file should contain your API keys for OpenAI and TogetherAI. The format of the `.env` file should be as follows and is provided as an example in `.env.example`:

```
OPENAI_API_KEY=your_openai_api_key
TOGETHERAI_API_KEY=your_togetherai_api_key
```

## Install Required Packages

To install the required Python packages used in this experiment run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Command Line Usage

```bash
Usage:
  python3 ./file - [arguments]

Arguments:
  --metadata (required)            File path to our synthesized error dataset metadata file containing class labels utilized and their descriptions
  --data (required)                File path to our synthesized error dataset file with gaze data and corresponding transcriptions on Chest X-Ray images
  --results (optional)             File path to preexisting results output file generated by the Python file
```
