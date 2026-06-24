import pandas as pd
import torch
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from langchain_community.tools import DuckDuckGoSearchRun

#df = pd.read_csv("datasets/dataset_test.csv")
#df = pd.read_csv("datasets/dataset_sample.csv")
df = pd.read_csv("datasets/dataset_clean.csv")

#test_df = df.sample(n=100, random_state=42).reset_index(drop=True)

search = DuckDuckGoSearchRun(max_results=1)
generator = pipeline("text-generation", device=0, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.float16)

system_prompt = (
    "You are an expert linguistic professor. Your task is to classify a given text as either 'Fact' or 'Opinion'.\n\n"
    "To ensure accuracy, you MUST analyze the text step-by-step before making a decision. "
    "Focus your analysis on: verifiability, subjective/emotional words, and objective entities or dates.\n"
    "CRITICAL WARNING: You are provided with external web search context. Just because a web search found information "
    "related to the topic does NOT mean the text is a Fact! Even highly discussed opinions will generate search results. "
    "If the text expresses a personal view, emotional judgment, or uses subjective words, it is an OPINION.\n\n"
    "You must follow this exact output format:\n"
    "Reasoning: [Your explanation in 3 sentences here]\n"
    "Classification: ['Fact' or 'Opinion']"
)


predictions = []
y_true = df["Classification"].tolist()
for i, row in df.iterrows():
  #user_text = f"Text to analyze: {row['Content']}"
  try:
    raw_context = search.run(row['Content'])
    if len(raw_context) > 400:
      context = raw_context[:400]
    else:
      context = raw_context
  except Exception as e:
    context = "No external context available."

  user_text = (
    f"External Context: {context}\n\n"
    f"Text to analyze: {row['Content']}"
  )

  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_text}
  ]

  output = generator(messages, max_new_tokens=300, do_sample=False)
  model_response = output[0]['generated_text'][-1]['content']
  #print(model_response)

  if "fact" in model_response.lower()[-40:]:
      pred_label = "Fact"
  elif "opinion" in model_response.lower()[-40:]:
      pred_label = "Opinion"
  else:
      pred_label = "none"

  predictions.append(pred_label)
  print("textindex", i+1, " | model prediction:", pred_label)

score = accuracy_score(y_true, predictions)

print("------------------------------")
print("chain of thought with RAG result")
print(f"Accuracy: {score:.2%}")

print("Confusion Matrix:")
print(confusion_matrix(y_true, predictions, labels=["Fact", "Opinion", "none"]))

"""
------------------------------
chain of thought and RAG together result
Accuracy: 71.00%
Confusion Matrix:
[[36 13  3]
 [10 35  3]
 [ 0  0  0]]
"""
