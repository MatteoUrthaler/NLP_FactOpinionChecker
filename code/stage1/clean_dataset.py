
import pandas as pd

df = pd.read_csv("datasets/dataset.csv")

df['Classification'] = df['Classification'].str.lower()

df.to_csv('datasets/dataset.csv', index=False)