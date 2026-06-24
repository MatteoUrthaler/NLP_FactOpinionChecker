import pandas as pd
import torch
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from langchain_community.tools import DuckDuckGoSearchRun

#df = pd.read_csv("datasets/dataset_test.csv")
#df = pd.read_csv("datasets/dataset_sample.csv")
df = pd.read_csv("datasets/dataset_clean.csv")

#test_df = df.sample(n=100, random_state=42).reset_index(drop=True)

search = DuckDuckGoSearchRun(max_results=1)
generator = pipeline("text-generation", device=0, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.float16)

system_prompt = (
  "Please classify a given text as either 'Fact' or 'Opinion'.\n"
  "You are provided with external context from a web search to help you verify the statement.\n\n"
  "You must follow this exact output format:\n"
  "Classification: [Strictly either 'Fact' or 'Opinion']"
)

predictions = []
y_true = df["Classification"].tolist()
for i, row in df.iterrows():

  #user_text = f"Text to analyze: {row['Content']}"
  try:
    raw_context = search.run(row['Content'])
    if len(raw_context) > 500:
      context = raw_context[:500]
    else:
      context = raw_context
  except Exception as e:
    context = "No external context available."

  user_text = (
    f"External Context:\n{context}\n\n"
    f"Text to analyze: {row['Content']}"
  )

  #print(len(context))
  #print(user_text)

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
print("RAG result")
print(f"Accuracy: {score:.2%}")

print("Confusion Matrix:")
print(confusion_matrix(y_true, predictions, labels=["Fact", "Opinion", "none"]))


"""
------------------------------
RAG result
Accuracy: 57.89%
Confusion Matrix:
[[424  19   0]
 [341  71   0]
 [  0   0   0]]
"""