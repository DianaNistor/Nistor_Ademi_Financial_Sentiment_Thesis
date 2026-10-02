from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "test_dataset.csv"
FEW_SHOT_PATH = PROJECT_ROOT / "dataset" / "few_shot_examples.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "full_results.csv"

# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

MODELS = [
    "gpt-5-mini",
    "deepseek-flash",
]

CONDITIONS = [
    "DP_ZERO",
    "DP_FEW",
    "COT_ZERO",
    "COT_FEW",
]

VALID_LABELS = {"positive", "neutral", "negative"}

EXPECTED_SENTENCES = 2255
EXPECTED_REQUESTS = EXPECTED_SENTENCES * len(MODELS) * len(CONDITIONS)

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0

# ---------------------------------------------------------------------
# Import frozen prompt functions
# ---------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prompts import (  # noqa: E402
    direct_zero_shot,
    direct_few_shot,
    cot_zero_shot,
    cot_few_shot,
)

# ---------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------

OUTPUT_FIELDS = [
    "timestamp_utc",
    "id",
    "sentence",
    "true_label",
    "model",
    "prompt_condition",
    "request_key",
    "predicted_label",
    "parse_status",
    "correct",
    "raw_response",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "error",
]


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def parse_prediction(raw_response: str):
    """
    Parse a model response into one of the three valid labels.

    Parsing order:
      1. FINAL: <label>
      2. exact label response
      3. exactly one unique label appearing in the response
      4. otherwise unparseable
    """

    if raw_response is None:
        return None, "empty_response"

    text = raw_response.strip().lower()

    if not text:
        return None, "empty_response"

    final_matches = re.findall(
        r"\bfinal\s*:\s*(positive|neutral|negative)\b",
        text,
    )

    if final_matches:
        return final_matches[-1], "parsed_final"

    if text in VALID_LABELS:
        return text, "parsed_exact"

    matches = re.findall(
        r"\b(positive|neutral|negative)\b",
        text,
    )

    unique_matches = set(matches)

    if len(unique_matches) == 1:
        return matches[0], "parsed_unique_label"

    return None, "unparseable"


# ---------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------

def extract_usage(usage):
    """
    Return:

        input_tokens
        output_tokens
        total_tokens

    Supports both:
      - OpenAI Responses API usage
      - OpenAI-compatible Chat Completions usage
    """

    if usage is None:
        return None, None, None

    # Object-style access
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    # DeepSeek / Chat Completions naming
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)

    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)

    if total_tokens is None:
        total_tokens = getattr(usage, "total_tokens", None)

    # Dictionary fallback
    if isinstance(usage, dict):
        input_tokens = usage.get(
            "input_tokens",
            usage.get("prompt_tokens", input_tokens),
        )
        output_tokens = usage.get(
            "output_tokens",
            usage.get("completion_tokens", output_tokens),
        )
        total_tokens = usage.get("total_tokens", total_tokens)

    return input_tokens, output_tokens, total_tokens


# ---------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------

def create_clients():
    """
    Create the two API clients.

    OpenAI:
        gpt-5-mini via Responses API

    DeepSeek:
        deepseek-flash via OpenAI-compatible Chat Completions API
    """

    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not available in the environment."
        )

    if not deepseek_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not available in the environment."
        )

    clients = {
        "gpt-5-mini": OpenAI(
            api_key=openai_key,
        ),
        "deepseek-flash": OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com",
        ),
    }

    return clients


# ---------------------------------------------------------------------
# API request functions
# ---------------------------------------------------------------------

def call_openai(client, prompt: str):
    """
    OpenAI Responses API call.

    The response object is returned so that both text and usage
    information can be extracted.
    """

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    raw_response = response.output_text

    return raw_response, response


def call_deepseek(client, prompt: str):
    """
    DeepSeek OpenAI-compatible Chat Completions API call.
    """

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    raw_response = response.choices[0].message.content

    return raw_response, response


def call_model(client, model: str, prompt: str):
    if model == "gpt-5-mini":
        return call_openai(client, prompt)

    if model == "deepseek-flash":
        return call_deepseek(client, prompt)

    raise ValueError(f"Unknown model: {model}")


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def build_prompt(
    sentence: str,
    condition: str,
    few_shot_examples: pd.DataFrame,
) -> str:

    if condition == "DP_ZERO":
        return direct_zero_shot(sentence)

    if condition == "DP_FEW":
        return direct_few_shot(
            sentence,
            few_shot_examples,
        )

    if condition == "COT_ZERO":
        return cot_zero_shot(sentence)

    if condition == "COT_FEW":
        return cot_few_shot(
            sentence,
            few_shot_examples,
        )

    raise ValueError(
        f"Unknown prompt condition: {condition}"
    )


# ---------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------

def execute_with_retries(
    client,
    model: str,
    prompt: str,
):
    """
    Execute one API request with exponential backoff.

    Returns:

        raw_response
        response_object
        error_message

    On success:
        error_message = None

    After all retries fail:
        raw_response = None
        response_object = None
        error_message = str(exception)
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            raw_response, response = call_model(
                client,
                model,
                prompt,
            )

            return raw_response, response, None

        except Exception as exc:
            last_error = exc

            print(
                f"  API error for {model} "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:
                delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))

                print(
                    f"  Retrying in {delay:.1f} seconds..."
                )

                time.sleep(delay)

    return None, None, str(last_error)


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def load_existing_results():
    """
    Load existing full results for resumability.

    Returns:
        dataframe
        set of completed request keys
    """

    if not RESULTS_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_FIELDS), set()

    df = pd.read_csv(
        RESULTS_PATH,
        encoding="utf-8-sig",
    )

    if df.empty:
        return df, set()

    required_columns = {
        "id",
        "model",
        "prompt_condition",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            "Existing full_results.csv is missing required "
            f"columns: {sorted(missing)}"
        )

    # A request counts as completed only when it has a prediction
    # or an API error has been recorded.
    #
    # This allows genuinely interrupted/incomplete rows to be retried
    # if they were ever written without a terminal status.
    completed = set()

    for row in df.itertuples(index=False):
        request_key = (
            str(row.id),
            str(row.model),
            str(row.prompt_condition),
        )

        prediction = getattr(row, "predicted_label", None)
        error = getattr(row, "error", None)

        prediction_is_valid = (
            pd.notna(prediction)
            and str(prediction).strip() != ""
        )

        error_is_recorded = (
            pd.notna(error)
            and str(error).strip() != ""
        )

        if prediction_is_valid or error_is_recorded:
            completed.add(request_key)

    return df, completed


def initialize_results_file():
    """
    Create the results CSV with the correct header if it does not exist.
    """

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not RESULTS_PATH.exists():
        with RESULTS_PATH.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=OUTPUT_FIELDS,
            )
            writer.writeheader()


def append_result(row: dict):
    """
    Append one completed request to the results CSV and flush it
    immediately.

    This makes the full experiment resumable even if the process
    stops unexpectedly.
    """

    initialize_results_file()

    with RESULTS_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writerow(row)
        f.flush()

        # Force the data to disk where supported.
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------

def validate_inputs(
    test_df: pd.DataFrame,
    few_shot_df: pd.DataFrame,
):
    """
    Validate the experimental inputs before any API request is made.
    """

    required_test_columns = {
        "id",
        "sentence",
        "label",
    }

    missing_test = required_test_columns - set(
        test_df.columns
    )

    if missing_test:
        raise RuntimeError(
            "test_dataset.csv is missing columns: "
            f"{sorted(missing_test)}"
        )

    required_few_shot_columns = {
        "id",
        "sentence",
        "label",
    }

    missing_few = required_few_shot_columns - set(
        few_shot_df.columns
    )

    if missing_few:
        raise RuntimeError(
            "few_shot_examples.csv is missing columns: "
            f"{sorted(missing_few)}"
        )

    # Check IDs
    if test_df["id"].duplicated().any():
        duplicated_ids = (
            test_df.loc[
                test_df["id"].duplicated(),
                "id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate test IDs detected: "
            f"{duplicated_ids[:10]}"
        )

    if few_shot_df["id"].duplicated().any():
        duplicated_ids = (
            few_shot_df.loc[
                few_shot_df["id"].duplicated(),
                "id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate few-shot IDs detected: "
            f"{duplicated_ids[:10]}"
        )

    # Check test/few-shot ID leakage
    test_ids = set(test_df["id"])
    few_ids = set(few_shot_df["id"])

    id_overlap = test_ids & few_ids

    if id_overlap:
        raise RuntimeError(
            "Test/few-shot ID overlap detected: "
            f"{sorted(id_overlap)[:10]}"
        )

    # Check sentence leakage as an additional safeguard
    test_sentences = set(test_df["sentence"])
    few_sentences = set(few_shot_df["sentence"])

    sentence_overlap = (
        test_sentences & few_sentences
    )

    if sentence_overlap:
        raise RuntimeError(
            "Test/few-shot sentence overlap detected. "
            f"Number of overlapping sentences: "
            f"{len(sentence_overlap)}"
        )

    # Validate labels
    invalid_test_labels = (
        set(test_df["label"].astype(str).str.lower())
        - VALID_LABELS
    )

    if invalid_test_labels:
        raise RuntimeError(
            "Invalid test labels detected: "
            f"{sorted(invalid_test_labels)}"
        )

    invalid_few_labels = (
        set(few_shot_df["label"].astype(str).str.lower())
        - VALID_LABELS
    )

    if invalid_few_labels:
        raise RuntimeError(
            "Invalid few-shot labels detected: "
            f"{sorted(invalid_few_labels)}"
        )

    # Confirm expected dataset size
    if len(test_df) != EXPECTED_SENTENCES:
        raise RuntimeError(
            f"Expected {EXPECTED_SENTENCES} test sentences, "
            f"but found {len(test_df)}."
        )

    # Confirm expected few-shot size
    if len(few_shot_df) != 9:
        raise RuntimeError(
            f"Expected 9 few-shot examples, "
            f"but found {len(few_shot_df)}."
        )


# ---------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------

def build_tasks(test_df: pd.DataFrame):
    """
    Generate all expected experimental request keys.

    The task order is:

        model
        condition
        sentence

    Each key is:

        (id, model, prompt_condition)
    """

    tasks = []

    for model in MODELS:
        for condition in CONDITIONS:
            for row in test_df.itertuples(index=False):

                request_key = (
                    str(row.id),
                    model,
                    condition,
                )

                tasks.append(
                    {
                        "id": str(row.id),
                        "sentence": str(row.sentence),
                        "true_label": str(row.label).lower(),
                        "model": model,
                        "prompt_condition": condition,
                        "request_key": "|".join(request_key),
                        "_key_tuple": request_key,
                    }
                )

    return tasks


# ---------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------

def run_dry_run(
    test_df: pd.DataFrame,
    few_shot_df: pd.DataFrame,
):
    print("=" * 70)
    print("FULL EXPERIMENT DRY RUN")
    print("=" * 70)

    print()
    print(f"Test sentences:       {len(test_df)}")
    print(f"Models:               {len(MODELS)}")
    print(f"Conditions:           {len(CONDITIONS)}")
    print()

    print("Models:")
    for model in MODELS:
        print(f"  {model}")

    print()
    print("Conditions:")
    for condition in CONDITIONS:
        print(f"  {condition}")

    print()

    tasks = build_tasks(test_df)

    expected = (
        len(test_df)
        * len(MODELS)
        * len(CONDITIONS)
    )

    unique_keys = {
        task["_key_tuple"]
        for task in tasks
    }

    print(f"Expected requests:    {expected}")
    print(f"Generated tasks:       {len(tasks)}")
    print(f"Unique request keys:   {len(unique_keys)}")
    print()

    # Verify exact expected count
    if expected != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Expected-request calculation mismatch: "
            f"{expected} != {EXPECTED_REQUESTS}"
        )

    # Verify generated task count
    if len(tasks) != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Generated task count mismatch: "
            f"{len(tasks)} != {EXPECTED_REQUESTS}"
        )

    # Verify unique keys
    if len(unique_keys) != EXPECTED_REQUESTS:
        raise RuntimeError(
            "Duplicate request keys detected in dry run."
        )

    # Condition/model counts
    counts = {}

    for task in tasks:
        key = (
            task["model"],
            task["prompt_condition"],
        )

        counts[key] = counts.get(key, 0) + 1

    print("Requests per model/condition:")

    for model in MODELS:
        for condition in CONDITIONS:
            count = counts[(model, condition)]

            print(
                f"  {model:17s} "
                f"{condition:10s} "
                f"{count}"
            )

            if count != EXPECTED_SENTENCES:
                raise RuntimeError(
                    f"Unexpected count for "
                    f"{model}/{condition}: {count}"
                )

    print()
    print("Few-shot examples:", len(few_shot_df))
    print()

    # Construct every prompt without making API calls.
    # This verifies that all four frozen prompt functions work
    # against the actual full dataset.
    prompt_count = 0

    for condition in CONDITIONS:
        for row in test_df.itertuples(index=False):
            _ = build_prompt(
                sentence=str(row.sentence),
                condition=condition,
                few_shot_examples=few_shot_df,
            )

            prompt_count += 1

    expected_prompt_count = (
        len(test_df) * len(CONDITIONS)
    )

    if prompt_count != expected_prompt_count:
        raise RuntimeError(
            "Prompt construction count mismatch."
        )

    print(
        f"Prompts successfully constructed: "
        f"{prompt_count}"
    )

    print()
    print("API calls made:       0")
    print()

    print("=" * 70)
    print("DRY RUN PASSED")
    print("=" * 70)


# ---------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------

def run_full_experiment(
    test_df: pd.DataFrame,
    few_shot_df: pd.DataFrame,
):
    print("=" * 70)
    print("FULL FINANCIAL PHRASEBANK EXPERIMENT")
    print("=" * 70)

    print()
    print(f"Test sentences:       {len(test_df)}")
    print(f"Models:               {len(MODELS)}")
    print(f"Conditions:           {len(CONDITIONS)}")
    print(f"Expected requests:    {EXPECTED_REQUESTS}")
    print()

    # Load existing results for resumability.
    existing_df, completed_keys = (
        load_existing_results()
    )

    if not existing_df.empty:
        print(
            f"Existing result rows: "
            f"{len(existing_df)}"
        )

    print(
        f"Completed request keys: "
        f"{len(completed_keys)}"
    )

    remaining = (
        EXPECTED_REQUESTS
        - len(completed_keys)
    )

    print(
        f"Remaining requests: "
        f"{remaining}"
    )

    print()

    if remaining == 0:
        print(
            "All expected requests are already present."
        )
        print(
            f"Results file: {RESULTS_PATH}"
        )
        return

    # Create API clients only when the actual experiment starts.
    clients = create_clients()

    tasks = build_tasks(test_df)

    # Final integrity check before making API calls.
    all_keys = {
        task["_key_tuple"]
        for task in tasks
    }

    if len(all_keys) != EXPECTED_REQUESTS:
        raise RuntimeError(
            "Full task list contains duplicate request keys."
        )

    request_number = 0
    skipped = 0
    completed_this_run = 0
    successful = 0
    unparseable = 0
    api_errors = 0
    correct = 0

    total_tasks = len(tasks)

    for task in tasks:

        key_tuple = task["_key_tuple"]

        # Resume logic
        if key_tuple in completed_keys:
            skipped += 1
            continue

        request_number += 1

        model = task["model"]
        condition = task["prompt_condition"]
        sentence = task["sentence"]
        true_label = task["true_label"]

        print(
            f"[{request_number}/{remaining}] "
            f"{model} | {condition} | {task['id']}"
        )

        prompt = build_prompt(
            sentence=sentence,
            condition=condition,
            few_shot_examples=few_shot_df,
        )

        raw_response, response, error = (
            execute_with_retries(
                client=clients[model],
                model=model,
                prompt=prompt,
            )
        )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        input_tokens = None
        output_tokens = None
        total_tokens = None

        predicted_label = None
        parse_status = None
        is_correct = False

        if response is not None:
            usage = getattr(
                response,
                "usage",
                None,
            )

            (
                input_tokens,
                output_tokens,
                total_tokens,
            ) = extract_usage(usage)

        if raw_response is not None:
            (
                predicted_label,
                parse_status,
            ) = parse_prediction(
                raw_response
            )

            if predicted_label is not None:
                successful += 1

                is_correct = (
                    predicted_label
                    == true_label
                )

                if is_correct:
                    correct += 1

            else:
                unparseable += 1

        else:
            parse_status = "api_error"
            api_errors += 1

        result_row = {
            "timestamp_utc": timestamp,
            "id": task["id"],
            "sentence": sentence,
            "true_label": true_label,
            "model": model,
            "prompt_condition": condition,
            "request_key": task["request_key"],
            "predicted_label": predicted_label,
            "parse_status": parse_status,
            "correct": is_correct,
            "raw_response": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error": error,
        }

        append_result(result_row)

        completed_keys.add(key_tuple)
        completed_this_run += 1

        if raw_response is not None:
            print(
                f"  prediction={predicted_label} "
                f"status={parse_status} "
                f"correct={is_correct}"
            )
        else:
            print(
                f"  API error recorded: {error}"
            )

        print(
            f"  Progress: "
            f"{len(completed_keys)}/{total_tasks}"
        )

    print()
    print("=" * 70)
    print("FULL EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Expected requests:       "
        f"{EXPECTED_REQUESTS}"
    )

    print(
        f"Unique completed keys:   "
        f"{len(completed_keys)}"
    )

    print(
        f"Requests executed now:   "
        f"{completed_this_run}"
    )

    print(
        f"Requests skipped/resumed:{skipped}"
    )

    print(
        f"Valid predictions:       "
        f"{successful}"
    )

    print(
        f"Unparseable responses:   "
        f"{unparseable}"
    )

    print(
        f"API errors:              "
        f"{api_errors}"
    )

    print(
        f"Correct predictions:     "
        f"{correct}"
    )

    if successful > 0:
        print(
            f"Descriptive accuracy:    "
            f"{correct / successful:.4f}"
        )

    print()
    print(
        f"Results saved to:\n"
        f"  {RESULTS_PATH}"
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Financial PhraseBank "
            "LLM sentiment experiment."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the full task structure and "
            "construct prompts without making API calls."
        ),
    )

    args = parser.parse_args()

    print(
        f"Project root:\n  {PROJECT_ROOT}"
    )

    print(
        f"Test dataset:\n  {DATASET_PATH}"
    )

    print(
        f"Few-shot examples:\n  {FEW_SHOT_PATH}"
    )

    print(
        f"Results file:\n  {RESULTS_PATH}"
    )

    print()

    # Load datasets
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{DATASET_PATH}"
        )

    if not FEW_SHOT_PATH.exists():
        raise FileNotFoundError(
            "Few-shot examples not found:\n"
            f"{FEW_SHOT_PATH}"
        )

    test_df = pd.read_csv(
        DATASET_PATH,
        encoding="utf-8-sig",
    )

    few_shot_df = pd.read_csv(
        FEW_SHOT_PATH,
        encoding="utf-8-sig",
    )

    # Normalize labels while preserving sentences.
    test_df["label"] = (
        test_df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    few_shot_df["label"] = (
        few_shot_df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    validate_inputs(
        test_df,
        few_shot_df,
    )

    print(
        f"Loaded {len(test_df)} test sentences."
    )

    print(
        f"Loaded {len(few_shot_df)} few-shot examples."
    )

    print()

    if args.dry_run:
        run_dry_run(
            test_df,
            few_shot_df,
        )
        return

    run_full_experiment(
        test_df,
        few_shot_df,
    )


if __name__ == "__main__":
    main()
    