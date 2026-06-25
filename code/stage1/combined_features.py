import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys

from pos_tagger import extract_pos_features
from NER import extract_ner_features
from emotion_analysis import extract_emotion_features
from error_analysis import extract_error_features
from readability_analysis import extract_readability_features

from pos_guesser import FEATURE_COLS as POS_COLS
from NER_guesser import FEATURE_COLS as NER_COLS
from emotion_guesser import FEATURE_COLS as EMOTION_COLS
from error_guesser import FEATURE_COLS as ERROR_COLS
from readability_guesser import FEATURE_COLS as READABILITY_COLS


INPUT_DATASET = "datasets/dataset_clean.csv"
N_SPLITS = 5
RANDOM_STATE = 69

USE_POS = True
USE_NER = True
USE_EMOTION = True
USE_ERROR = False
USE_READABILITY = False

pos_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
ner_weight = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
emotion_weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

WEIGHTS = {
    "pos": pos_weight,
    "ner": ner_weight,
    "emotion": emotion_weight,
    "error": 1.0,
    "readability": 1.0,
}


def extract_combined_features(text):
    combined = {}

    if USE_POS:
        combined.update(extract_pos_features(text))

    if USE_NER:
        combined.update(extract_ner_features(text))

    if USE_EMOTION:
        combined.update(extract_emotion_features(text))

    if USE_ERROR:
        combined.update(extract_error_features(text))

    if USE_READABILITY:
        combined.update(extract_readability_features(text))

    return combined

def get_feature_cols():
    cols = []

    if USE_POS:
        cols.extend(POS_COLS)

    if USE_NER:
        cols.extend(NER_COLS)

    if USE_EMOTION:
        cols.extend(EMOTION_COLS)

    if USE_ERROR:
        cols.extend(ERROR_COLS)

    if USE_READABILITY:
        cols.extend(READABILITY_COLS)

    return cols

def get_weight_vector(feature_cols):
    weights = []

    for col in feature_cols:
        if col in POS_COLS:
            weights.append(WEIGHTS["pos"])
        elif col in NER_COLS:
            weights.append(WEIGHTS["ner"])
        elif col in EMOTION_COLS:
            weights.append(WEIGHTS["emotion"])
        elif col in ERROR_COLS:
            weights.append(WEIGHTS["error"])
        elif col in READABILITY_COLS:
            weights.append(WEIGHTS["readability"])
        else:
            weights.append(1.0)

    return np.array(weights, dtype=float)

def predict_text(text, fact_mean, opinion_mean, feature_cols, weight_vector):
    features = extract_combined_features(text)
    x = np.array([features[col] for col in feature_cols], dtype=float)

    weighted_x = x * weight_vector
    weighted_fact_mean = fact_mean * weight_vector
    weighted_opinion_mean = opinion_mean * weight_vector

    dist_fact = np.linalg.norm(weighted_x - weighted_fact_mean)
    dist_opinion = np.linalg.norm(weighted_x - weighted_opinion_mean)

    return "Fact" if dist_fact < dist_opinion else "Opinion"


df = pd.read_csv(INPUT_DATASET)
df = df.reset_index(drop=True)

x = df["Content"]
y = df["Classification"]

feature_cols = get_feature_cols()
weight_vector = get_weight_vector(feature_cols)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []

for fold_number, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_features = train_df["Content"].apply(extract_combined_features)
    train_features_df = pd.DataFrame(train_features.tolist())

    train_result_df = pd.concat(
        [train_features_df.reset_index(drop=True), train_df["Classification"].reset_index(drop=True)],
        axis=1
    )

    means = train_result_df.groupby("Classification")[feature_cols].mean()

    fact_mean = means.loc["Fact", feature_cols].values.astype(float)
    opinion_mean = means.loc["Opinion", feature_cols].values.astype(float)

    correct_count = 0
    total = len(test_df)

    for _, row in test_df.iterrows():
        prediction = predict_text(
            row["Content"],
            fact_mean,
            opinion_mean,
            feature_cols,
            weight_vector
        )

        if prediction == row["Classification"]:
            correct_count += 1

    accuracy = (correct_count / total) * 100
    fold_accuracies.append(accuracy)

    print(f"Fold {fold_number}: {accuracy:.2f}% ({correct_count}/{total})")

mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\nFeature columns: {feature_cols}")
print(f"Weights: {weight_vector.tolist()}")
print(f"Fold accuracies: {fold_accuracies}")
print(f"Mean accuracy: {mean_accuracy:.2f}%")
print(f"Standard deviation: {std_accuracy:.2f}")