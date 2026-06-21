import pandas as pd
import numpy as np
import spacy # type: ignore
from collections import Counter

from pos_tagger import extract_pos_features
import sys

input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.csv"
means_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_pos_means.csv"


SAMPLE_SIZE = 50

model = spacy.load("en_core_web_sm")

FEATURE_COLS = [
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "propernouns_ratio",
    "adjective_noun_ratio",
]

df = pd.read_csv(input_csv)

means = pd.read_csv(means_csv, index_col=0)

fact_mean = means.loc["fact", FEATURE_COLS].values.astype(float)
opinion_mean = means.loc["opinion", FEATURE_COLS].values.astype(float)


sample_df = df.sample(n=SAMPLE_SIZE, random_state=69).copy()

def predict_text(text):
    features = extract_pos_features(text)
    x = np.array([features[col] for col in FEATURE_COLS], dtype=float)

    dist_fact = np.linalg.norm(x - fact_mean)
    dist_opinion = np.linalg.norm(x - opinion_mean)

    prediction = "fact" if dist_fact < dist_opinion else "opinion"
    return prediction

correct_count = 0
total = len(sample_df)

for _, row in sample_df.iterrows():
    prediction = predict_text(row["Content"])

    if prediction == row["Classification"]:
        correct_count += 1

accuracy = (correct_count / total) * 100

print(f"\nSample size: {SAMPLE_SIZE}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct guesses: {correct_count} / {SAMPLE_SIZE}")
