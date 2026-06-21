import pandas as pd
import spacy
import language_tool_python
import os


OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



model = spacy.load("en_core_web_sm")

# pip install language-tool-python
tool = language_tool_python.LanguageTool("en-US")


df = pd.read_csv("datasets/dataset.csv")


def extract_error_features(text):
    matches = tool.check(text)

    tokens = [t for t in model(text) if not t.is_punct and not t.is_space]
    total_tokens = len(tokens)

    spelling_count = 0
    grammar_count = 0

    for match in matches:
        issue_type = match.rule_issue_type

        if issue_type == "misspelling":
            spelling_count += 1
        else:
            grammar_count += 1

    total_error_count = len(matches)

    return {
        "error_ratio": total_error_count / total_tokens,
        "spelling_error_ratio": spelling_count / total_tokens,
        "grammar_error_ratio": grammar_count / total_tokens,
    }


error_features = df["Content"].apply(extract_error_features)
error_df = pd.DataFrame(error_features.tolist())

print(error_df.head())

result_df = pd.concat([error_df, df["Classification"]], axis=1)
print(result_df.head())


feature_cols = [
    "error_ratio",
    "spelling_error_ratio",
    "grammar_error_ratio",
]

means = result_df.groupby("Classification")[feature_cols].mean()

print("\nAverages:")
print(means)


means.to_csv(OUTPUT_DIR + "/fact_opinion_error_means.csv", index=True)