import pandas as pd
import os
import textstat

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

textstat.set_lang("en")


df = pd.read_csv("datasets/dataset.csv")

def extract_readability_features(text):

    avg_sentence_length = textstat.avg_sentence_length(text)
    avg_syllables_per_word = textstat.avg_syllables_per_word(text)
    
    difficult_words = textstat.difficult_words(text)
    total_words = textstat.lexicon_count(text, removepunct=True)
    difficult_words_ratio = (difficult_words / total_words)
    
    flesch_ease = textstat.flesch_reading_ease(text)
    
    return {
        # average number of words per sentence
        "avg_sentence_length": avg_sentence_length,
        # average number of syllables per word
        "avg_syllables_per_word": avg_syllables_per_word,
        # ratio of difficult words (3 or more syllables)
        "difficult_words_ratio": difficult_words_ratio,
        # Flesch Reading Ease score (can be 0 to 100)
        # ==> 100 easy to read | 0 difficult to read (academic)
        "flesch_reading_ease": flesch_ease
    }


readability_features = df["Content"].apply(extract_readability_features)
readability_df = pd.DataFrame(readability_features.tolist())

result_df = pd.concat([readability_df, df["Classification"]], axis=1)
print(result_df.head())




feature_cols = [
    "avg_sentence_length",
    "avg_syllables_per_word",
    "difficult_words_ratio",
    "flesch_reading_ease"
]

means = result_df.groupby("Classification")[feature_cols].mean()

print("\nAverages for readability:")
print(means)

means.to_csv(OUTPUT_DIR + "/fact_opinion_readability_means.csv", index=True)