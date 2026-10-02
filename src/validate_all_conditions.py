import os
import pandas as pd

from openai import OpenAI

from prompts import (
    direct_zero_shot,
    direct_few_shot,
    cot_zero_shot,
    cot_few_shot,
)


TEST_PATH = "dataset/test_dataset.csv"
FEW_SHOT_PATH = "dataset/few_shot_examples.csv"


def main():
    print("=" * 70)
    print("8-CONDITION API VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Check API keys
    # ------------------------------------------------------------

    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    print("OpenAI API key: found")
    print("DeepSeek API key: found")

    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------

    test_dataset = pd.read_csv(TEST_PATH)
    few_shot = pd.read_csv(FEW_SHOT_PATH)

    row = test_dataset.iloc[0]

    sentence_id = row["id"]
    sentence = row["sentence"]
    true_label = row["label"]

    print()
    print("Test ID:", sentence_id)
    print("Sentence:", sentence)
    print("True label:", true_label)

    # ------------------------------------------------------------
    # Create API clients
    # ------------------------------------------------------------

    openai_client = OpenAI(
        api_key=openai_key
    )

    deepseek_client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    # ------------------------------------------------------------
    # Define four experimental conditions
    # ------------------------------------------------------------

    conditions = {
        "DP_ZERO": lambda: direct_zero_shot(sentence),
        "DP_FEW": lambda: direct_few_shot(sentence, few_shot),
        "COT_ZERO": lambda: cot_zero_shot(sentence),
        "COT_FEW": lambda: cot_few_shot(sentence, few_shot),
    }

    # ------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("OPENAI")
    print("=" * 70)

    for condition_name, prompt_function in conditions.items():

        print()
        print("-" * 70)
        print("Condition:", condition_name)
        print("-" * 70)

        prompt = prompt_function()

        print("Sending request to OpenAI...")

        response = openai_client.responses.create(
            model="gpt-5-mini",
            input=prompt,
        )

        model_response = response.output_text

        print("OpenAI response:")
        print(model_response)

        print()
        print("Token usage:")
        print(response.usage)

    # ------------------------------------------------------------
    # DeepSeek
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DEEPSEEK")
    print("=" * 70)

    for condition_name, prompt_function in conditions.items():

        print()
        print("-" * 70)
        print("Condition:", condition_name)
        print("-" * 70)

        prompt = prompt_function()

        print("Sending request to DeepSeek...")

        response = deepseek_client.chat.completions.create(
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

        print("DeepSeek response:")
        print(model_response)

        print()
        print("Token usage:")
        print(response.usage)

    print()
    print("=" * 70)
    print("8-CONDITION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    