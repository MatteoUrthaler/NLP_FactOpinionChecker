import pandas as pd
import spacy
import language_tool_python
import os
import sys

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model = spacy.load("en_core_web_sm")

# pip install language-tool-python
tool = language_tool_python.LanguageTool("en-US")



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

if __name__ == "__main__":

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_error_means.csv"

    df = pd.read_csv(input_csv)

    error_features = df["Content"].apply(extract_error_features)
    error_df = pd.DataFrame(error_features.tolist())

    result_df = pd.concat([error_df, df["Classification"]], axis=1)
    #print(result_df.head())


    feature_cols = [
        "error_ratio",
        "spelling_error_ratio",
        "grammar_error_ratio",
    ]

    means = result_df.groupby("Classification")[feature_cols].mean()

    #print("\nAverages:")
    #print(means)


    means.to_csv(output_csv, index=True)
print("error done")
