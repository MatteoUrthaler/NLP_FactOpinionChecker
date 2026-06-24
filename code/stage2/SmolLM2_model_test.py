import pandas as pd
import torch
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

#df = pd.read_csv("datasets/dataset_test.csv")
#df = pd.read_csv("datasets/dataset_sample.csv")
df = pd.read_csv("datasets/dataset_clean.csv")

generator = pipeline("text-generation", device=0, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.float16)

system_prompt = (
  "Please classify a given text as either 'Fact' or 'Opinion'.\n"
  "You must follow this exact output format:\n"
  "Classification: [Strictly either 'Fact' or 'Opinion']"
)

predictions = []
y_true = df["Classification"].tolist()
for i, row in df.iterrows():
  user_text = f"Text to analyze: {row['Content']}"
  
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_text}
  ]
  
  output = generator(messages, max_new_tokens=20, do_sample=False)
  model_response = output[0]['generated_text'][-1]['content']
  #print(model_response)
  
  if "fact" in model_response.lower():
      pred_label = "Fact"
  elif "opinion" in model_response.lower():
      pred_label = "Opinion"
  else:
      pred_label = "none"
  
  predictions.append(pred_label)
  print("textindex", i+1, " | model prediction:", pred_label)


score = accuracy_score(y_true, predictions)

print("------------------------------")
print("zero shot result")
print(f"Accuracy: {score:.2%}")

print("Confusion Matrix:")
print(confusion_matrix(y_true, predictions, labels=["Fact", "Opinion", "none"]))

"""
------------------------------
zero shot result
Accuracy: 66.43%
Confusion Matrix:
[[417  26   0]
 [261 151   0]
 [  0   0   0]]
------------------------------
 """