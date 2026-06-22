import pandas as pd
import spacy # type: ignore
from collections import Counter
import os
import sys

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


model = spacy.load("en_core_web_sm")

def extract_ner_features(text):
    doc = model(text)

    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)

    if total == 0:
        return {
            "entity_ratio": 0.0,
            "person_ratio": 0.0,
            "organization_ratio": 0.0,
            "gpe_ratio": 0.0,
            "location_ratio": 0.0,
            "date_ratio": 0.0,
        }

    entities = list(doc.ents)
    total_entities = len(entities)

    entity_counts = Counter(ent.label_ for ent in entities)

    person_count = entity_counts.get("PERSON", 0)
    organization_count = entity_counts.get("ORG", 0)
    gpe_count = entity_counts.get("GPE", 0)
    location_count = entity_counts.get("LOC", 0)
    date_count = entity_counts.get("DATE", 0)

    entity_ratio = total_entities / total
    person_ratio = person_count / total
    org_ratio = organization_count / total
    gpe_ratio = gpe_count / total
    location_ratio = location_count / total
    date_ratio = date_count / total

    return {
        "entity_ratio": entity_ratio,
        "person_ratio": person_ratio,
        "organization_ratio": org_ratio,
        "gpe_ratio": gpe_ratio,
        "location_ratio": location_ratio,
        "date_ratio": date_ratio,
    }

if __name__ == "__main__":

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_ner_means.csv"

    df = pd.read_csv(input_csv)

    ner_features = df["Content"].apply(extract_ner_features)
    ner_df = pd.DataFrame(ner_features.tolist())

    result_df = pd.concat([ner_df, df["Classification"]], axis=1)
    #print(result_df.head())

    feature_cols = [
        "entity_ratio",
        "person_ratio",
        "organization_ratio",
        "gpe_ratio",
        "location_ratio",
        "date_ratio",
    ]

    means = result_df.groupby("Classification")[feature_cols].mean()

    #print("\nAverages:")
    #print(means)

    means.to_csv(output_csv, index=True)

print("ner done")