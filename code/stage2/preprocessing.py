import pandas as pd
import unicodedata
import re

def normalize(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00A0", " ")

    text = re.sub(r"(?<=[A-Za-z])[�ÿ\uFFFD](?=[A-Za-z])", "'", text)
    text = re.sub(r"[�ÿ\uFFFD]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def cjk(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))

def symbol_ratio(text):
    no_space = re.sub(r"\s+", "", text)
    if not no_space:
        return False

    ratio = sum(1 for ch in no_space if not ch.isalnum()) / len(no_space)
    return ratio > 0.3

def prompt_injection(text):
    t = str(text).lower()

    prompts = [
        "[system prompt]",
        "note to ai:",
        "ai system instruction:",
        "note to llms:",
    ]

    return any(m in t for m in prompts)

def should_drop(text):
    return prompt_injection(text) or cjk(text) or symbol_ratio(text)

def capitalize(x):
    if x == "fact":
        return "Fact"
    if x == "opinion":
        return "Opinion"
    return x

def run_preprocessing(input_file, output_file):

    df = pd.read_csv(input_file)

    if input_file == "datasets/dataset.csv":
        df['Classification'] = df['Classification'].str.lower()
        df['Classification'] = df['Classification'].apply(capitalize)

    df["Content"] = df["Content"].apply(normalize)
    df["drop_row"] = df["Content"].apply(should_drop)

    df.reset_index(inplace=True)
    df.rename(columns={"index": "original_index"}, inplace=True)

    df["original_index"] = range(1, len(df) + 1)

    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    run_preprocessing("datasets/dataset.csv", "datasets/dataset_clean.csv")
    run_preprocessing("datasets/validationset.csv", "datasets/validationset_clean.csv")