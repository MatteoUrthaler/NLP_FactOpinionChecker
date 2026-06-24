import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("datasets/dataset_clean.csv")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

predictions = []

for i, row in df.iterrows():
    result = classifier(row["Content"], candidate_labels=["Fact", "Opinion"])
    predictions.append(result["labels"][0] )
    print(i)


classification = df["Classification"].tolist()

print(f"Accuracy: {accuracy_score(classification, predictions):.2%}")
print("\nReport:")
print(classification_report(classification, predictions, target_names=["Fact", "Opinion"]))
print("\nConfusion Matrix:")
print(confusion_matrix(classification, predictions, labels=["Fact", "Opinion"]))

"""RESULTS:
Accuracy: 59.53%

Report:
              precision    recall  f1-score   support

        Fact       0.61      0.62      0.61       443
     Opinion       0.58      0.57      0.57       412

    accuracy                           0.60       855
   macro avg       0.59      0.59      0.59       855
weighted avg       0.60      0.60      0.60       855


Confusion Matrix:
[[275 168]
 [178 234]]"""