import pandas as pd
import re
import unicodedata

INPUT_FILE = "datasets/dataset.csv"
OUTPUT_FILE = "datasets/dataset_clean.csv"
TEXT_COL = "Content"

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")

    # Fix malformed apostrophe-like symbols between letters
    text = re.sub(r"(?<=[A-Za-z])[�ÿ\uFFFD](?=[A-Za-z])", "'", text)

    # Replace remaining bad replacement symbols with spaces
    text = re.sub(r"[�ÿ\uFFFD]+", " ", text)

    # Keep ellipses and normal punctuation, just normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def has_cjk(text):
    return re.search(r"[\u4e00-\u9fff]", text) is not None

def symbol_ratio(text):
    no_space = re.sub(r"\s+", "", text)
    if not no_space:
        return 0.0

    symbol_count = sum(
        1 for ch in no_space
        if not ch.isalnum()
    )
    return symbol_count / len(no_space)

def should_drop(text):
    if has_cjk(text):
        return True
    if symbol_ratio(text) > 0.30:
        return True
    return False

df = pd.read_csv(INPUT_FILE)

df[TEXT_COL] = df[TEXT_COL].apply(normalize_text)
df["drop_row"] = df[TEXT_COL].apply(should_drop)

df.reset_index(inplace=True)
df.rename(columns={"index": "original_index"}, inplace=True)

df.to_csv(OUTPUT_FILE, index=False)