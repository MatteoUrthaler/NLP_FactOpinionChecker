import pandas as pd
import os

from pos_tagger import extract_pos_features
from NER import extract_ner_features
from emotion_analysis import extract_emotion_features
from readability_analysis import extract_readability_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("PLS WAIT - MAY TAKE UP TO 3 MINS")

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


df = pd.read_csv("datasets/dataset.csv")

def extract_all_features(text):
    pos_features = extract_pos_features(text)
    ner_features = extract_ner_features(text)
    emotion_features = extract_emotion_features(text)
    readability_features = extract_readability_features(text)

    return pos_features | ner_features | emotion_features | readability_features


combined_features = df["Content"].apply(extract_all_features)
features_df = pd.DataFrame(combined_features.tolist())

result_df = pd.concat([features_df, df["Classification"]], axis=1)
print(result_df.head())


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
    "sentiment_neu",
    "subjectivity",
    "difficult_words_ratio",
    "flesch_reading_ease"
]

X = result_df[FEATURE_COLS]
y = result_df["Classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

models = {
    "Support Vector Machine (Linear)": SVC(kernel="linear", C=1.0, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions) * 100
    
    print("-------------------------------------------------")
    print(f"ML Model: {name}")
    print(f"Final Accuracy: {acc:.2f}%\n")
    print("-------------------------------------------------")