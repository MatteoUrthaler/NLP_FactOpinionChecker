# NLP Project - Group 6

Automated pipeline for classifying text data into **Fact** or **Opinion**. The pipeline handles the entire workflow: text preprocessing (cleaning, filtering characters, symbol ratios, and prompt injections), feature extraction, model training, and evaluation by classifying the validationset.csv.

## Setup

This project was developed using Python 3.14.

Follow these steps to set up the environment and install required libs:

Create and activate a virtual environment:

   ```bash
   # Create venv
   python -m venv .venv

   # Activate venv (Windows)
   .venv\Scripts\activate

   # Activate venv (Mac/Linux)
   source .venv/bin/activate
   ```

Install requirements from root directory:
   ```bash
   # Install requirements
   pip install -r requirements.txt
   ```

## Run Pipeline Stage 1

The pipeline is fully integrated so that you only need to run one single script.
Please execute from root directory:

   ```bash
   python code/stage1/verdict.py
   ```

After the pipeline completes successfully, all generated artifacts will be stored in the output/ directory. The final predictions for the validation set can be found in *group6_classifications_1.csv*.


## Run Pipeline Stage 2

The pipeline is fully integrated so that you only need to run one single script.
Please execute from root directory:

   ```bash
   python code/stage2/verdict.py
   ```

After the pipeline completes successfully the final predictions for the validation set can be found in *group6_classifications_2.csv*.