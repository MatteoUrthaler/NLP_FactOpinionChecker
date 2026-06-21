import os
import re
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


N_SPLITS = 5
RANDOM_STATE = 69

INPUT_DATASET = "datasets/dataset.csv"
WORK_DIR = "output/kfold_pos"

os.makedirs(WORK_DIR, exist_ok=True)

df = pd.read_csv(INPUT_DATASET)
df = df.reset_index(drop=True)

x = df["Content"]
y = df["Classification"]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []

for fold_number, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_path = f"{WORK_DIR}/train_fold_{fold_number}.csv"
    test_path = f"{WORK_DIR}/test_fold_{fold_number}.csv"
    means_path = f"{WORK_DIR}/means_fold_{fold_number}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    subprocess.run(
        ["python3", "code/stage1/pos_tagger.py", train_path, means_path],
        check=True
    )

    result = subprocess.run(
        ["python3", "code/stage1/pos_guesser.py", test_path, means_path],
        check=True,
        capture_output=True,
        text=True
    )

    match = re.search(r"Accuracy:\s*([0-9.]+)%", result.stdout)
    if match:
        accuracy = float(match.group(1))
        fold_accuracies.append(accuracy)
    else:
        print(f"Missing accuracy for fold {fold_number}")

mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"Accuracies: {fold_accuracies}")
print(f"Mean accuracy: {mean_accuracy:.2f}%")
print(f"Standard deviation: {std_accuracy:.2f}")