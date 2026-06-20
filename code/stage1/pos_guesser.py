import pandas as pd
import numpy as np
import spacy
from collections import Counter


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

def extract_pos_features(text):
    doc = model(text)

    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)

    counts = Counter(t.pos_ for t in tokens)

    noun_ratio = counts.get("NOUN", 0) / total
    verb_ratio = counts.get("VERB", 0) / total
    adjective_ratio = counts.get("ADJ", 0)  / total
    adverb_ratio = counts.get("ADV", 0)  / total
    pronoun_ratio = counts.get("PRON", 0) / total
    propernouns_ratio = counts.get("PROPN", 0) / total

    noun_count = counts.get("NOUN", 0)
    adjective_noun_ratio = counts.get("ADJ", 0) / noun_count if noun_count > 0 else 0

    return {
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adjective_ratio": adjective_ratio,
        "adverb_ratio": adverb_ratio,
        "pronoun_ratio": pronoun_ratio,
        "propernouns_ratio": propernouns_ratio,
        "adjective_noun_ratio": adjective_noun_ratio,
    }

df = pd.read_csv("datasets/dataset.csv")

means = pd.read_csv("output/fact_opinion_means.csv", index_col=0)

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
