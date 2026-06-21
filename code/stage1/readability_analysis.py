import pandas as pd
import os
import textstat
import sys

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

textstat.set_lang("en")


def extract_readability_features(text):

    avg_sentence_length = textstat.avg_sentence_length(text)
    avg_syllables_per_word = textstat.avg_syllables_per_word(text)
    
    difficult_words = textstat.difficult_words(text)
    total_words = textstat.lexicon_count(text, removepunct=True)
    difficult_words_ratio = (difficult_words / total_words)
    
    flesch_ease = textstat.flesch_reading_ease(text)

    if flesch_ease < 0:
        flesch_ease = 0

    if flesch_ease > 100:
        flesch_ease = 100

    flesch_min = 0
    flesch_max = 100
    flesch_scaled = (flesch_ease - flesch_min) / (flesch_max - flesch_min)
    
    flesch_scaled = max(0.0, min(1.0, flesch_scaled))
    
    return {
        # average number of words per sentence
        #"avg_sentence_length": avg_sentence_length,
        # average number of syllables per word
        #"avg_syllables_per_word": avg_syllables_per_word,
        # ratio of difficult words (3 or more syllables)
        "difficult_words_ratio": difficult_words_ratio,
        # Flesch Reading Ease score (can be 0 to 100)
        # ==> 100 easy to read | 0 difficult to read (academic)
        "flesch_reading_ease": flesch_scaled
    }

if __name__ == "__main__":

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_readability_means.csv"


    df = pd.read_csv(input_csv)

    readability_features = df["Content"].apply(extract_readability_features)
    readability_df = pd.DataFrame(readability_features.tolist())

    result_df = pd.concat([readability_df, df["Classification"]], axis=1)
    #print(result_df.head())

    feature_cols = [
        #"avg_sentence_length",
        #"avg_syllables_per_word",
        "difficult_words_ratio",
        "flesch_reading_ease"
    ]

    means = result_df.groupby("Classification")[feature_cols].mean()

    #print("\nAverages for readability:")
    #print(means)

    means.to_csv(output_csv, index=True)

print("readability done")