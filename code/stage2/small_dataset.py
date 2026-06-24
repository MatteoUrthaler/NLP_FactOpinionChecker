import pandas as pd

df = pd.read_csv("datasets/dataset_clean.csv")

facts   = df[df["Classification"] == "Fact"].sample(15, random_state=69)
opinions = df[df["Classification"] == "Opinion"].sample(15, random_state=69)

sample = pd.concat([facts, opinions]).sample(frac=1, random_state=69).reset_index(drop=True)
sample.to_csv("datasets/dataset_sample.csv", index=False)