import os
import pandas as pd
from openai import OpenAI

from prompts import direct_zero_shot


# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")


# Create OpenAI client
client = OpenAI(api_key=api_key)


# Load test dataset
test_dataset = pd.read_csv("dataset/test_dataset.csv")

# Take only ONE test sentence
row = test_dataset.iloc[0]

sentence = row["sentence"]
true_label = row["label"]


# Create our existing zero-shot prompt
prompt = direct_zero_shot(sentence)


print("=" * 70)
print("TEST SENTENCE")
print("=" * 70)

print(sentence)
print("True label:", true_label)


print("\n" + "=" * 70)
print("PROMPT SENT TO OPENAI")
print("=" * 70)

print(prompt)


# Send ONE request to OpenAI
response = client.responses.create(
    model="gpt-5-mini",
    input=prompt,
)


# Get the model's response
model_response = response.output_text


print("\n" + "=" * 70)
print("OPENAI RESPONSE")
print("=" * 70)

print(model_response)
