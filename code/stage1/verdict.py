import pandas as pd
import pickle
from sklearn.svm import SVC

from pos_tagger import extract_pos_features
from NER import extract_ner_features
from emotion_analysis import extract_emotion_features
from readability_analysis import extract_readability_features

from preprocessing import run_preprocessing


RECOMPILE = True


DATASET_TRAIN_RAW = "datasets/dataset.csv"
DATASET_VALIDATE_RAW = "datasets/validationset.csv"
DATASET_TRAIN = "datasets/dataset_clean.csv"
DATASET_VALIDATE = "datasets/validationset_clean.csv"
OUTPUT = "group6_classifications_1.csv"

MODEL = "output/verdict_model.pkl"
FEATURES = "output/verdict_features.pkl"

TEXT_COL = "Content"
VERDICT_COL = "Classification"
DROP_COL = "drop_row"

FEATURE_COLS = [
    # from pos
    "noun_ratio", "adverb_ratio", "pronoun_ratio", "propernouns_ratio",
    # from ner
    "entity_ratio", "date_ratio",
    # from emotion
    "sentiment_neu", "subjectivity",
    # from readability
    "flesch_reading_ease"
]

def extract_all_features(text):
    text = "" if pd.isna(text) else str(text)
    pos_features = extract_pos_features(text)
    ner_features = extract_ner_features(text)
    emotion_features = extract_emotion_features(text)
    readability_features = extract_readability_features(text)
    return pos_features | ner_features | emotion_features | readability_features

def build_feature_frame(df):
    combined_features = df[TEXT_COL].apply(extract_all_features)
    features_df = pd.DataFrame(combined_features.tolist())

    for col in FEATURE_COLS:
        if col not in features_df.columns:
            features_df[col] = 0.0

    return features_df[FEATURE_COLS].fillna(0)

def normalize_label(value):
    value = str(value).strip().lower()
    if value == "fact":
        return "Fact"
    if value == "opinion":
        return "Opinion"
    return value




if __name__ == "__main__":

    run_preprocessing(DATASET_TRAIN_RAW, DATASET_TRAIN)
    run_preprocessing(DATASET_VALIDATE_RAW, DATASET_VALIDATE)

    if RECOMPILE:
        df_train = pd.read_csv(DATASET_TRAIN)

        X_train = build_feature_frame(df_train)
        y_train = df_train[VERDICT_COL]

        model = SVC(kernel="linear", C=1.0, random_state=42)
        model.fit(X_train, y_train)

        with open(MODEL, "wb") as f:
            pickle.dump(model, f)

        with open(FEATURES, "wb") as f:
            pickle.dump(FEATURE_COLS, f)
    else:
        with open(MODEL, "rb") as f:
            model = pickle.load(f)

        with open(FEATURES, "rb") as f:
            FEATURE_COLS = pickle.load(f)

    df_val = pd.read_csv(DATASET_VALIDATE)

    x = build_feature_frame(df_val)
    preds = model.predict(x)

    if DROP_COL not in df_val.columns:
        print("run preprocessing first!")

    verdicts = []
    for i, pred in enumerate(preds):
        if df_val.iloc[i][DROP_COL] == True:
            verdicts.append("none")
        else:
            verdicts.append(normalize_label(pred))

    result_df = pd.DataFrame({
        "ID": range(1, len(df_val) + 1),
        "Verdict": verdicts
    })

    result_df.to_csv(OUTPUT, index=False)
    print("\nresult ready")