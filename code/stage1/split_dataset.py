import pandas as pd
import os


# Split ratios
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3


SPLIT_DIR = "../../datasets/split"
if not os.path.exists(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

TRAINING_PATH = SPLIT_DIR + "/train.csv"
TEST_PATH = SPLIT_DIR + "/test.csv"


df = pd.read_csv("../../datasets/dataset.csv")


df = df.sample(frac=1).reset_index(drop=True)

size = len(df)
train_end = int(TRAIN_RATIO * size)
test_end = int((TRAIN_RATIO + TEST_RATIO) * size)

train = df.iloc[:train_end]
test = df.iloc[train_end:test_end]


train.to_csv(TRAINING_PATH, index=False)
test.to_csv(TEST_PATH, index=False)