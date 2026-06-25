import pandas as pd
import os
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import sys

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

nltk.download('vader_lexicon', quiet=True)
nltk_sentiment_intensity_analyzer = SentimentIntensityAnalyzer()



def extract_emotion_features(text):
    
    vader_scores = nltk_sentiment_intensity_analyzer.polarity_scores(text)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    
    return {
        # ratio of neutral words
        "sentiment_neu": vader_scores["neu"],
        # subjectivity score
        # ==> 0 fully objective | 1 fully subjective
        "subjectivity": subjectivity

        # at one point we also had these features in mind:

        # we came to the conclusion that these were redundant information (already covered by others)
        # ratio of positive words
        # "sentiment_pos": vader_scores["pos"],
        # ratio of negative words
        # "sentiment_neg": vader_scores["neg"],
        
        # we replaced the compund by 'sentiment_neutral' to capture intensity without directional bias
        # emotional intensity score (can be -1 to 1)
        # ==> -1 extreme negative | 0 fully neutral | 1 extremely positive
        # "sentiment_compound": vader_scores["compound"],

    }

if __name__ == "__main__":

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset_clean.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "output/fact_opinion_emotion_means.csv"

    df = pd.read_csv(input_csv)
    emotion_features = df["Content"].apply(extract_emotion_features)
    emotion_df = pd.DataFrame(emotion_features.tolist())

    result_df = pd.concat([emotion_df, df["Classification"]], axis=1)
    #print(result_df.head())




    feature_cols = [
        #"sentiment_compound",
        #"sentiment_pos",
        #"sentiment_neg",
        "sentiment_neu",
        "subjectivity"
    ]

    means = result_df.groupby("Classification")[feature_cols].mean()

    #print("\nAverages for emotions and subjectivity:")
    #print(means)

    means.to_csv(output_csv, index=True)

print("emotion done")