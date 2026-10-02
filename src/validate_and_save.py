import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI

from prompts import (
    direct_zero_shot,
    direct_few_shot,
    cot_zero_shot,
    cot_few_shot,
)


TEST_PATH = Path("dataset/test_dataset.csv")
FEW_SHOT_PATH = Path("dataset/few_shot_examples.csv")
OUTPUT_PATH = Path("results/validation_results.csv")


VALID_LABELS = {"positive", "neutral", "negative"}


def parse_prediction(raw_response: str):
    """
    Extract one of the three allowed sentiment labels.

    For CoT responses, prefer the label appearing after FINAL:.
    For direct responses, accept a response containing exactly
    one valid label.

    Returns:
        predicted_label, parse_status
    """

    if raw_response is None:
        return None, "empty_response"

    text = raw_response.strip().lower()

    if not text:
        return None, "empty_response"

    # ------------------------------------------------------------
    # First: look for the explicit CoT final answer
    # ------------------------------------------------------------

    final_matches = re.findall(
        r"\bfinal\s*:\s*(positive|neutral|negative)\b",
        text,
    )

    if final_matches:
        return final_matches[-1], "parsed_final"

    # ------------------------------------------------------------
    # Second: accept a response that is exactly one label
    # ------------------------------------------------------------

    if text in VALID_LABELS:
        return text, "parsed_exact"

    # ------------------------------------------------------------
    # Third: detect an otherwise unambiguous label
    # ------------------------------------------------------------

    matches = re.findall(
        r"\b(positive|neutral|negative)\b",
        text,
    )

    unique_matches = set(matches)

    if len(unique_matches) == 1:
        return matches[0], "parsed_unique_label"

    # ------------------------------------------------------------
    # Otherwise, parsing failed
    # ------------------------------------------------------------

    return None, "unparseable"


def extract_usage(response):
    """
    Extract token usage from either OpenAI Responses API
    or DeepSeek Chat Completions API.
    """

    usage = getattr(response, "usage", None)

    if usage is None:
        return None, None, None, None

    input_tokens = getattr(usage, "input_tokens", None)

    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)

    output_tokens = getattr(usage, "output_tokens", None)

    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)

    total_tokens = getattr(usage, "total_tokens", None)

    reasoning_tokens = None

    output_details = getattr(usage, "output_tokens_details", None)

    if output_details is not None:
        reasoning_tokens = getattr(
            output_details,
            "reasoning_tokens",
            None,
        )

    completion_details = getattr(
        usage,
        "completion_tokens_details",
        None,
    )

    if reasoning_tokens is None and completion_details is not None:
        reasoning_tokens = getattr(
            completion_details,
            "reasoning_tokens",
            None,
        )

    return (
        input_tokens,
        output_tokens,
        reasoning_tokens,
        total_tokens,
    )


def save_results(results):
    """
    Save validation results to CSV.
    Each validation run starts with a clean file.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_PATH,
        mode="w",
        header=True,
        index=False,
    )


def main():

    print("=" * 70)
    print("VALIDATION RUNNER")
    print("=" * 70)

    # ------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------

    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------

    test_dataset = pd.read_csv(TEST_PATH)
    few_shot = pd.read_csv(FEW_SHOT_PATH)

    row = test_dataset.iloc[0]

    sentence_id = row["id"]
    sentence = row["sentence"]
    true_label = row["label"]

    print("Test ID:", sentence_id)
    print("True label:", true_label)
    print()

    # ------------------------------------------------------------
    # API clients
    # ------------------------------------------------------------

    openai_client = OpenAI(
        api_key=openai_key,
    )

    deepseek_client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    # ------------------------------------------------------------
    # Prompt conditions
    # ------------------------------------------------------------

    conditions = {
        "DP_ZERO": lambda: direct_zero_shot(sentence),
        "DP_FEW": lambda: direct_few_shot(sentence, few_shot),
        "COT_ZERO": lambda: cot_zero_shot(sentence),
        "COT_FEW": lambda: cot_few_shot(sentence, few_shot),
    }

    results = []

    # ------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------

    for condition_name, prompt_function in conditions.items():

        print(f"OpenAI | {condition_name}")

        prompt = prompt_function()

        timestamp = datetime.now(timezone.utc).isoformat()

        try:

            response = openai_client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            )

            raw_response = response.output_text

            predicted_label, parse_status = parse_prediction(
                raw_response
            )

            (
                input_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
            ) = extract_usage(response)

            correct = (
                predicted_label == true_label
                if predicted_label is not None
                else None
            )

            result = {
                "id": sentence_id,
                "sentence": sentence,
                "true_label": true_label,
                "model": "gpt-5-mini",
                "prompt_condition": condition_name,
                "raw_response": raw_response,
                "predicted_label": predicted_label,
                "parse_status": parse_status,
                "correct": correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "timestamp_utc": timestamp,
                "error": None,
            }

        except Exception as e:

            result = {
                "id": sentence_id,
                "sentence": sentence,
                "true_label": true_label,
                "model": "gpt-5-mini",
                "prompt_condition": condition_name,
                "raw_response": None,
                "predicted_label": None,
                "parse_status": "api_error",
                "correct": None,
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
                "timestamp_utc": timestamp,
                "error": repr(e),
            }

        results.append(result)

        # Save immediately after every request
        save_results(results)

        print(
            "Prediction:",
            result["predicted_label"],
            "| Status:",
            result["parse_status"],
        )

    # ------------------------------------------------------------
    # DeepSeek
    # ------------------------------------------------------------

    for condition_name, prompt_function in conditions.items():

        print(f"DeepSeek | {condition_name}")

        prompt = prompt_function()

        timestamp = datetime.now(timezone.utc).isoformat()

        try:

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

            raw_response = response.choices[0].message.content

            predicted_label, parse_status = parse_prediction(
                raw_response
            )

            (
                input_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
            ) = extract_usage(response)

            correct = (
                predicted_label == true_label
                if predicted_label is not None
                else None
            )

            result = {
                "id": sentence_id,
                "sentence": sentence,
                "true_label": true_label,
                "model": "deepseek-flash",
                "prompt_condition": condition_name,
                "raw_response": raw_response,
                "predicted_label": predicted_label,
                "parse_status": parse_status,
                "correct": correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "timestamp_utc": timestamp,
                "error": None,
            }

        except Exception as e:

            result = {
                "id": sentence_id,
                "sentence": sentence,
                "true_label": true_label,
                "model": "deepseek-flash",
                "prompt_condition": condition_name,
                "raw_response": None,
                "predicted_label": None,
                "parse_status": "api_error",
                "correct": None,
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
                "timestamp_utc": timestamp,
                "error": repr(e),
            }

        results.append(result)

        # Save immediately after every request
        save_results(results)

        print(
            "Prediction:",
            result["predicted_label"],
            "| Status:",
            result["parse_status"],
        )

    print()
    print("=" * 70)
    print("VALIDATION RUN COMPLETE")
    print("=" * 70)
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
