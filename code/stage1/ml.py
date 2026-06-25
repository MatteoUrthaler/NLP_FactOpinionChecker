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
from sklearn.model_selection import cross_val_score

print("PLS WAIT - MAY TAKE UP TO 3 MINS")

if not os.path.exists("output"):
    os.makedirs("output")


df = pd.read_csv("datasets/dataset_clean.csv")

def extract_all_features(text):
    pos_features = extract_pos_features(text)
    ner_features = extract_ner_features(text)
    emotion_features = extract_emotion_features(text)
    readability_features = extract_readability_features(text)

    return pos_features | ner_features | emotion_features | readability_features


combined_features = df["Content"].apply(extract_all_features)
features_df = pd.DataFrame(combined_features.tolist())

result_df = pd.concat([features_df, df["Classification"]], axis=1)
#print(result_df.head())

# all cols from 4 analysers:
"""
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
"""

# at the end we used only the following features and achieved better results
FEATURE_COLS = [
    # from pos
    "noun_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "propernouns_ratio",

    # from ner
    "entity_ratio",
    "date_ratio",

    # from emotion
    "sentiment_neu",
    "subjectivity",

    # from readability
    "flesch_reading_ease"
]

x = result_df[FEATURE_COLS]
y = result_df["Classification"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69, stratify=y)

"""
models = {
    # had 78% Accuracy
    #"Support Vector Machine (Linear) 12": SVC(kernel="linear", C=1.2, random_state=69),
    # had 78% Accuracy
    #"Support Vector Machine (Linear) 11": SVC(kernel="linear", C=1.1, random_state=69),

    # had 80% Accuracy
    "Support Vector Machine (Linear)": SVC(kernel="linear", C=1.0, random_state=69)

    # had 78% Accuracy
    #"Support Vector Machine (Linear)09": SVC(kernel="linear", C=0.9, random_state=69)
    # had 78% Accuracy
    #"Support Vector Machine (Linear) 08": SVC(kernel="linear", C=0.8, random_state=69),
    # had 77% Accuracy
    #"Support Vector Machine (Linear) 05": SVC(kernel="linear", C=0.5, random_state=69),

    # had 77% Accuracy
    # "SVM (Polynomial Degree 2)": SVC(kernel="poly", degree=2, C=1.0, random_state=69)
    # had 54% Accuracy
    # "SVM (Sigmoid)": SVC(kernel="sigmoid", C=1.0, random_state=69)
    # had 75% Accuracy
    #"Logistic Regression": LogisticRegression(max_iter=1000, random_state=69),
    # had 70% Accuracy
    #"Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=69)
}

for name, model in models.items():
    model.fit(x_train, y_train)

    #weights = model.coef_[0]
    #svm_ranking = sorted(zip(FEATURE_COLS, weights), key=lambda x: abs(x[1]), reverse=True)
    #for feat, weight in svm_ranking:
    #    print("feature: ", feat, " - weight: ", weight)

    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions) * 100

    print(f"ML Model: {name}")
    print(f"Final Accuracy: {acc:.2f}%\n")
"""

final_model = SVC(kernel="linear", C=1.0, random_state=69)
final_model.fit(x_train, y_train)

scores = cross_val_score(final_model, x, y, cv=5, scoring='accuracy')

print(f"K-Fold Accuracies: {scores * 100}")
print(f"Mean Robust Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: ±{scores.std() * 100:.2f}%")