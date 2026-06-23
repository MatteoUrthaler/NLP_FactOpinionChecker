import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("datasets/dataset_clean.csv")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1#0 for GPU
)

predictions = []

for i, row in df.iterrows():
    result = classifier(row["Content"], candidate_labels=["Fact", "Opinion"])
    predictions.append(result["labels"][0] )
    print(i)


classification = df["Classification"].tolist()

print(f"Accuracy: {accuracy_score(classification, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(classification, predictions, target_names=["Fact", "Opinion"]))
print("\nConfusion Matrix:")
print(confusion_matrix(classification, predictions, labels=["Fact", "Opinion"]))
