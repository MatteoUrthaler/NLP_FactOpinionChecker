import pandas as pd
import numpy as np
import spacy # type: ignore
from collections import Counter

from NER import extract_ner_features
from pos_tagger import extract_pos_features

SAMPLE_SIZE = 50


FEATURE_COLS = [
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "propernouns_ratio",
    "adjective_noun_ratio",
    "entity_ratio",
    "person_ratio",
    "organization_ratio",
    "gpe_ratio",
    "location_ratio",
    "date_ratio",
]

pos_means = pd.read_csv("output/fact_opinion_pos_means.csv", index_col=0)
ner_means = pd.read_csv("output/fact_opinion_ner_means.csv", index_col=0)
combined_means = pd.concat([pos_means, ner_means], axis=1)
combined_means.to_csv("output/fact_opinion_combined_means.csv", index=True)


means = pd.read_csv("output/fact_opinion_combined_means.csv", index_col=0)

fact_mean = means.loc["fact", FEATURE_COLS].values.astype(float)
opinion_mean = means.loc["opinion", FEATURE_COLS].values.astype(float)


def extract_combined_features(text):
    pos_features = extract_pos_features(text)
    ner_features = extract_ner_features(text)

    return pos_features | ner_features

def predict_text(text):
    features = extract_combined_features(text)
    x = np.array([features[col] for col in FEATURE_COLS], dtype=float)

    dist_fact = np.linalg.norm(x - fact_mean)
    dist_opinion = np.linalg.norm(x - opinion_mean)

    return "fact" if dist_fact < dist_opinion else "opinion"

df = pd.read_csv("datasets/dataset.csv")

sample_df = df.sample(n=SAMPLE_SIZE, random_state=69).copy()


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