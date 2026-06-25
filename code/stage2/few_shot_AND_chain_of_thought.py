import pandas as pd
import torch
import numpy as np
import sys
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#df = pd.read_csv("datasets/dataset_test.csv")
#df = pd.read_csv("datasets/dataset_sample.csv")
input_csv = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset_clean.csv"
output_csv = sys.argv[2] if len(sys.argv) > 2 else "output/few_and_chain.csv"
df = pd.read_csv(input_csv)

#test_df = df.sample(n=100, random_state=69).reset_index(drop=True)

generator = pipeline("text-generation", device=0, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.float16)

system_prompt = (
    "You are a precise and impartial text classification engine. Your task is to classify a given sentence as either a Fact or an Opinion.\n\n"
    "Definitions:\n"
    "- Fact: A statement that is objectively verifiable, can be proven true or false through evidence, data, or observation, and does not express personal feelings, beliefs, or judgments.\n"
    "- Opinion: A statement that expresses a personal belief, subjective judgment, preference, interpretation, or evaluation that cannot be objectively verified as universally true or false.\n\n"
    "Rules:\n"
    "1. Base your classification solely on the linguistic and semantic content of the sentence.\n"
    "2. You MUST analyze the text step-by-step under the 'Reasoning:' header before making a decision.\n"
    "3. Do not let hedging language or authoritative tone automatically determine your answer — evaluate the underlying claim.\n\n"
    "You must follow this exact output format strictly. Do not add asterisks to headers:\n"
    "Reasoning: Let's think step by step. [Your explanation in max 3 sentences]\n"
    "Classification: [Strictly either Fact or Opinion]"
)


predictions = []
for i, row in df.iterrows():

  user_text = (
      "Here are examples of the required analysis and output format:\n\n"
      
      "Text to analyze: The 2008 global financial crisis led to the loss of approximately 8.7 million jobs in the United States between 2008 and 2010.\n"
      "Reasoning: Let's think step by step. This statement contains specific, measurable data regarding job losses and a defined timeframe. These metrics can be objectively verified using official economic records. Therefore, it is a factual claim.\n"
      "Classification: Fact\n\n"

      "Text to analyze: The government's handling of the economic downturn in 2008 was irresponsible and caused unnecessary suffering for millions of working-class families.\n"
      "Reasoning: Let's think step by step. The terms 'irresponsible' and 'unnecessary suffering' are subjective value judgments and emotional characterizations. A government's performance cannot be universally proven as 'irresponsible' through raw data alone. Thus, it represents a viewpoint.\n"
      "Classification: Opinion\n\n"

      "Text to analyze: According to the World Health Organization, tobacco use is responsible for over 8 million deaths per year worldwide.\n"
      "Reasoning: Let's think step by step. This sentence references an official statistic provided by a specific organization. The accuracy of this report and the underlying death toll can be checked and verified against health data. It is an objective statement.\n"
      "Classification: Fact\n\n"

      "Text to analyze: Given the overwhelming evidence of its harmful effects, any government that continues to allow tobacco advertising is putting corporate profits above the health of its citizens.\n"
      "Reasoning: Let's think step by step. The statement makes a moral and political accusation about government motivations ('putting corporate profits above health'). This is an interpretation and a political evaluation, which cannot be scientifically proven true or false. It is an opinion.\n\n"

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
      predict = "Fact"
  elif "opinion" in model_response.lower()[-40:]:
      predict = "Opinion"
  else:
      predict = "none"

  predictions.append(predict)
  print("textindex", i+1, " | model prediction:", predict)

output_df = pd.DataFrame({
    "original_index": df["original_index"].tolist(),
    "Classification": predictions,
    "drop_row": df["drop_row"].tolist()
})
output_df.to_csv(output_csv, index=False)

if input_csv == "datasets/dataset_clean.csv":
  y_true = df["Classification"].tolist()
  score = accuracy_score(y_true, predictions)

  print("few shot + chain of thought result")
  print(f"Accuracy: {score:.2%}")
  print("Confusion Matrix:")
  print(confusion_matrix(y_true, predictions, labels=["Fact", "Opinion", "none"]))

"""
few shot + chain of thought result
Accuracy: 85.03%
Confusion Matrix:
[[424  18   1]
 [107 303   2]
 [  0   0   0]]
"""