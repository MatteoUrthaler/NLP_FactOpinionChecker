import pandas as pd
import numpy as np
from collections import Counter

from emotion_analysis import extract_emotion_features
import sys

SAMPLE_SIZE = 100

FEATURE_COLS = [
    #"sentiment_compound",
    #"sentiment_pos",
    #"sentiment_neg",
    "sentiment_neu",
    "subjectivity",
]

def predict_text(text):
    features = extract_emotion_features(text)
    x = np.array([features[col] for col in FEATURE_COLS], dtype=float)

    dist_fact = np.linalg.norm(x - fact_mean)
    dist_opinion = np.linalg.norm(x - opinion_mean)

    prediction = "fact" if dist_fact < dist_opinion else "opinion"
    return prediction

if __name__ == "__main__":

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.csv"
    means_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_emotion_means.csv"

    df = pd.read_csv(input_csv)

    means = pd.read_csv(means_csv, index_col=0)

    fact_mean = means.loc["fact", FEATURE_COLS].values.astype(float)
    opinion_mean = means.loc["opinion", FEATURE_COLS].values.astype(float)


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