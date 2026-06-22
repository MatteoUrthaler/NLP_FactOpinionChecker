import pandas as pd
import spacy # type: ignore
from collections import Counter
import os
import sys

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#python -m spacy download en_core_web_sm)
model = spacy.load("en_core_web_sm")

def extract_pos_features(text):
    doc = model(text)

    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)

    counts = Counter(t.pos_ for t in tokens)

    if total == 0:
        return {
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adjective_ratio": 0.0,
            "adverb_ratio": 0.0,
            "pronoun_ratio": 0.0,
            "propernouns_ratio": 0.0,
            "adjective_noun_ratio": 0.0,
        }

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

if __name__ == "__main__":

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_pos_means.csv"

    df = pd.read_csv(input_csv)

    pos_features = df["Content"].apply(extract_pos_features)
    pos_df = pd.DataFrame(pos_features.tolist())

    result_df = pd.concat([pos_df, df["Classification"]], axis=1)
    #print(result_df.head())

    feature_cols = [
        "noun_ratio",
        "verb_ratio",
        "adjective_ratio",
        "adverb_ratio",
        "pronoun_ratio",
        "propernouns_ratio",
        "adjective_noun_ratio",
    ]

    means = result_df.groupby("Classification")[feature_cols].mean()

    #print("\nAverages:")
    #print(means)

    means.to_csv(output_csv, index=True)

print("pos done")