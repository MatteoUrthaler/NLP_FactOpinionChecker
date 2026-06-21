import pandas as pd
import os
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

nltk.download('vader_lexicon')
nltk_sentiment_intensity_analyzer = SentimentIntensityAnalyzer()


df = pd.read_csv("datasets/dataset.csv")

def extract_emotion_features(text):
    
    vader_scores = nltk_sentiment_intensity_analyzer.polarity_scores(text)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    
    return {
        # emotional intensity score (can be -1 to 1)
        # ==> -1 extreme negative | 0 fully neutral | 1 extremely positive
        "sentiment_compound": vader_scores["compound"],
        # ratio of positive words
        "sentiment_pos": vader_scores["pos"],
        # ratio of negative words
        "sentiment_neg": vader_scores["neg"],
        # ratio of neutral words
        "sentiment_neu": vader_scores["neu"],
        # subjectivity score (can be 0 to 1)
        # ==> 0 fully objective | 1 fully subjective
        "subjectivity": subjectivity
    }


emotion_features = df["Content"].apply(extract_emotion_features)
emotion_df = pd.DataFrame(emotion_features.tolist())

result_df = pd.concat([emotion_df, df["Classification"]], axis=1)
print(result_df.head())




feature_cols = [
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "subjectivity"
]

means = result_df.groupby("Classification")[feature_cols].mean()

print("\nAverages for emotions and subjectivity:")
print(means)

means.to_csv(OUTPUT_DIR + "/fact_opinion_emotion_means.csv", index=True)