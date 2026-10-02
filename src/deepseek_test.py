import os
import pandas as pd
from openai import OpenAI

from prompts import direct_zero_shot


TEST_PATH = "dataset/test_dataset.csv"


def main():
    print("Starting DeepSeek API test...")

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is not set."
        )

    print("DeepSeek API key found.")

    test_df = pd.read_csv(TEST_PATH)

    row = test_df.iloc[0]

    sentence = row["sentence"]
    true_label = row["label"]

    prompt = direct_zero_shot(sentence)

    print("Test sentence loaded.")
    print("Sending request to DeepSeek...")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
    )

    model_response = response.choices[0].message.content

    print()
    print("=== DeepSeek API Test ===")
    print()
    print("Sentence:")
    print(sentence)
    print()
    print("True label:")
    print(true_label)
    print()
    print("Prompt:")
    print(prompt)
    print()
    print("DeepSeek response:")
    print(model_response)
    print()

    print("Token usage:")

    if response.usage:
        print(response.usage)
    else:
        print("No token usage information returned.")


if __name__ == "__main__":
    main()
    