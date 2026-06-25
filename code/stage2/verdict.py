import subprocess
import sys
import os
import pandas as pd
from preprocessing import run_preprocessing

DATASET_VALIDATE_RAW = "datasets/validationset.csv"
DATASET_VALIDATE     = "datasets/validationset_clean.csv"
OUTPUT               = "group6_classifications_2.csv"
STAGE2_SCRIPT        = "code/stage2/few_shot_AND_chain_of_thought.py"
STAGE2_OUTPUT        = "output/few_and_chain_val.csv"

RECOMPILE = False

DROP_COL  = "drop_row"
TEXT_COL  = "Content"

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
            [sys.executable, STAGE2_SCRIPT, DATASET_VALIDATE, STAGE2_OUTPUT],
            check=True
        )

    # 3. load stage2 predictions
    preds_df = pd.read_csv(STAGE2_OUTPUT)   # columns: original_index (or index), Classification
    df_val   = pd.read_csv(DATASET_VALIDATE)

    # stage2 script writes predictions in row order — align by position
    predictions = preds_df["Classification"].tolist()

    if DROP_COL not in df_val.columns:
        print("run preprocessing first!")

    verdicts = []
    for i, pred in enumerate(predictions):
        if df_val.iloc[i][DROP_COL] == True:
            verdicts.append("none")
        else:
            verdicts.append(normalize_label(pred))

    result_df = pd.DataFrame({
        "ID":      range(1, len(df_val) + 1),
        "Verdict": verdicts
    })

    os.makedirs(os.path.dirname(OUTPUT) if os.path.dirname(OUTPUT) else ".", exist_ok=True)
    result_df.to_csv(OUTPUT, index=False)
    print("\nresult ready")