import subprocess
import sys
import pandas as pd
from preprocessing import run_preprocessing

DATASET_VALIDATE_RAW = "datasets/validationset.csv"
DATASET_VALIDATE = "datasets/validationset_clean.csv"
OUTPUT = "group6_classifications_2.csv"
FEW_AND_CHAIN = "code/stage2/few_shot_AND_chain_of_thought.py"
FAC_RESULTS = "output/few_and_chain_val.csv"

RECOMPILE = True

def normalize_label(value):
    value = str(value).strip().lower()
    if value == "fact":
        return "Fact"
    if value == "opinion":
        return "Opinion"
    return value

if __name__ == "__main__":

    if RECOMPILE:
        run_preprocessing(DATASET_VALIDATE_RAW, DATASET_VALIDATE)
        subprocess.run(
            [sys.executable, FEW_AND_CHAIN, DATASET_VALIDATE, FAC_RESULTS],
            check=True
        )

    df = pd.read_csv(FAC_RESULTS)

    predictions = df["Classification"].tolist()

    if "drop_row" not in df.columns:
        print("error: no preprocessing")

    verdicts = []
    for i, pred in enumerate(predictions):
        if df.iloc[i]["drop_row"] == True:
            verdicts.append("none")
        else:
            verdicts.append(normalize_label(pred))

    result_df = pd.DataFrame({
        "ID":      range(1, len(df) + 1),
        "Verdict": verdicts
    })

    result_df.to_csv(OUTPUT, index=False)
    print("\nresult ready")