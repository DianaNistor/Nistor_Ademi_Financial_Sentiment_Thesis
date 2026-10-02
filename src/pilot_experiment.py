import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from prompts import (  # noqa: E402
    direct_zero_shot,
    direct_few_shot,
    cot_zero_shot,
    cot_few_shot,
)

from validate_and_save import (  # noqa: E402
    parse_prediction,
    extract_usage,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PILOT_PATH = PROJECT_ROOT / "dataset" / "pilot_dataset.csv"
FEW_SHOT_PATH = PROJECT_ROOT / "dataset" / "few_shot_examples.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "pilot_results.csv"

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

VALID_LABELS = {
    "positive",
    "neutral",
    "negative",
}


# ---------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------

def create_clients():
    clients = {}

    if os.getenv("OPENAI_API_KEY"):
        clients["gpt-5-mini"] = OpenAI()

    if os.getenv("DEEPSEEK_API_KEY"):
        clients["deepseek-flash"] = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    return clients


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def build_prompt(condition, sentence, few_shot):
    if condition == "DP_ZERO":
        return direct_zero_shot(sentence)

    if condition == "DP_FEW":
        return direct_few_shot(sentence, few_shot)

    if condition == "COT_ZERO":
        return cot_zero_shot(sentence)

    if condition == "COT_FEW":
        return cot_few_shot(sentence, few_shot)

    raise ValueError(
        f"Unknown prompt condition: {condition}"
    )


# ---------------------------------------------------------------------
# API request
# ---------------------------------------------------------------------

def run_request(client, model, prompt):
    """
    Send one request to the selected model.

    Returns:
        raw_response,
        usage_dict,
        error_message
    """

    try:
        if model == "gpt-5-mini":

            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            )

            raw_response = response.output_text

        elif model == "deepseek-flash":

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

            raw_response = response.choices[0].message.content

        else:
            raise ValueError(
                f"Unsupported model: {model}"
            )

        # extract_usage() from validate_and_save.py returns
        # four values:
        #
        # input_tokens
        # output_tokens
        # reasoning_tokens
        # total_tokens

        usage_values = extract_usage(response)

        usage = {
            "input_tokens": usage_values[0],
            "output_tokens": usage_values[1],
            "reasoning_tokens": usage_values[2],
            "total_tokens": usage_values[3],
        }

        return raw_response, usage, None

    except Exception as exc:

        return None, {}, str(exc)


# ---------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------

def build_tasks(pilot_df):
    """
    Build all experiment requests.

    20 sentences
    × 2 models
    × 4 conditions
    = 160 requests
    """

    tasks = []

    for model in MODELS:

        for row in pilot_df.itertuples(index=False):

            for condition in CONDITIONS:

                tasks.append(
                    {
                        "id": row.id,
                        "sentence": row.sentence,
                        "true_label": row.label,
                        "model": model,
                        "prompt_condition": condition,
                    }
                )

    return tasks


# ---------------------------------------------------------------------
# Unique request key
# ---------------------------------------------------------------------

def make_key(record):
    return (
        str(record["id"]),
        str(record["model"]),
        str(record["prompt_condition"]),
    )


# ---------------------------------------------------------------------
# Existing results
# ---------------------------------------------------------------------

def load_existing_results():
    """
    Load previously saved results.

    The pilot is resumable:
    - completed requests are skipped
    - API-error requests are retried
    """

    if not OUTPUT_PATH.exists():
        return {}

    try:
        df = pd.read_csv(
            OUTPUT_PATH,
            encoding="utf-8-sig",
        )

    except Exception as exc:

        print(
            f"Warning: could not read existing results: {exc}"
        )

        print(
            "Starting with an empty result set."
        )

        return {}

    records_by_key = {}

    for record in df.to_dict("records"):

        try:
            key = make_key(record)
            records_by_key[key] = record

        except KeyError:
            continue

    return records_by_key


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_results(records_by_key):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = list(
        records_by_key.values()
    )

    df = pd.DataFrame(records)

    if not df.empty:

        df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )


# ---------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------

def run_dry_run(tasks):

    print()
    print("=" * 70)
    print("PILOT DRY RUN")
    print("=" * 70)

    print(
        f"Pilot dataset:       {PILOT_PATH}"
    )

    print(
        f"Few-shot examples:   {FEW_SHOT_PATH}"
    )

    print(
        f"Output file:         {OUTPUT_PATH}"
    )

    print()

    unique_ids = {
        task["id"]
        for task in tasks
    }

    unique_keys = {
        make_key(task)
        for task in tasks
    }

    print(
        f"Pilot sentences:     {len(unique_ids)}"
    )

    print(
        f"Models:              {len(MODELS)}"
    )

    print(
        f"Conditions:          {len(CONDITIONS)}"
    )

    print(
        f"Expected requests:   {len(tasks)}"
    )

    print(
        f"Unique request keys: {len(unique_keys)}"
    )

    expected = 20 * 2 * 4

    if len(tasks) != expected:

        raise RuntimeError(
            f"Expected {expected} tasks, "
            f"but generated {len(tasks)}."
        )

    if len(unique_keys) != expected:

        raise RuntimeError(
            f"Expected {expected} unique request keys, "
            f"but found {len(unique_keys)}."
        )

    print()

    print("Models:")

    for model in MODELS:
        print(f"  - {model}")

    print()

    print("Conditions:")

    for condition in CONDITIONS:
        print(f"  - {condition}")

    print()

    print("First 10 planned requests:")

    for number, task in enumerate(
        tasks[:10],
        start=1,
    ):

        print(
            f"  {number:02d}. "
            f"{task['id']} | "
            f"{task['model']} | "
            f"{task['prompt_condition']}"
        )

    print()

    print("Last 5 planned requests:")

    start_number = len(tasks) - 4

    for number, task in enumerate(
        tasks[-5:],
        start=start_number,
    ):

        print(
            f"  {number:03d}. "
            f"{task['id']} | "
            f"{task['model']} | "
            f"{task['prompt_condition']}"
        )

    print()

    print("=" * 70)
    print("DRY RUN PASSED")
    print("No API requests were made.")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------
# Actual pilot
# ---------------------------------------------------------------------

def run_pilot(tasks, few_shot, clients):

    records_by_key = load_existing_results()

    total_tasks = len(tasks)

    for task_number, task in enumerate(
        tasks,
        start=1,
    ):

        key = make_key(task)

        existing = records_by_key.get(key)

        # -------------------------------------------------------------
        # Skip completed requests.
        #
        # API errors are NOT considered completed because they should
        # be retried.
        # -------------------------------------------------------------

        if existing is not None:

            parse_status = str(
                existing.get(
                    "parse_status",
                    "",
                )
            )

            if parse_status != "api_error":

                print(
                    f"[{task_number:03d}/{total_tasks}] "
                    f"SKIP  "
                    f"{task['id']} | "
                    f"{task['model']} | "
                    f"{task['prompt_condition']}"
                )

                continue

        # -------------------------------------------------------------
        # Build prompt
        # -------------------------------------------------------------

        prompt = build_prompt(
            task["prompt_condition"],
            task["sentence"],
            few_shot,
        )

        print()

        print(
            f"[{task_number:03d}/{total_tasks}] "
            f"RUN   "
            f"{task['id']} | "
            f"{task['model']} | "
            f"{task['prompt_condition']}"
        )

        # -------------------------------------------------------------
        # API request
        # -------------------------------------------------------------

        client = clients.get(
            task["model"]
        )

        if client is None:

            raw_response = None
            usage = {}

            error = (
                f"No API client available for "
                f"{task['model']}."
            )

        else:

            raw_response, usage, error = run_request(
                client=client,
                model=task["model"],
                prompt=prompt,
            )

        # -------------------------------------------------------------
        # Parse response
        # -------------------------------------------------------------

        if error is not None:

            predicted_label = None
            parse_status = "api_error"
            correct = False

        else:

            predicted_label, parse_status = (
                parse_prediction(
                    raw_response
                )
            )

            correct = (
                predicted_label == task["true_label"]
                if predicted_label in VALID_LABELS
                else False
            )

        # -------------------------------------------------------------
        # Build result
        # -------------------------------------------------------------

        result = {
            "id": task["id"],
            "sentence": task["sentence"],
            "true_label": task["true_label"],
            "model": task["model"],
            "prompt_condition": task["prompt_condition"],
            "raw_response": raw_response,
            "predicted_label": predicted_label,
            "parse_status": parse_status,
            "correct": correct,
            "input_tokens": usage.get(
                "input_tokens"
            ),
            "output_tokens": usage.get(
                "output_tokens"
            ),
            "reasoning_tokens": usage.get(
                "reasoning_tokens"
            ),
            "total_tokens": usage.get(
                "total_tokens"
            ),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "error": error,
        }

        # -------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------

        records_by_key[key] = result

        # -------------------------------------------------------------
        # Save immediately after every request
        # -------------------------------------------------------------

        save_results(
            records_by_key
        )

        # -------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------

        if error is not None:

            print(
                f"       ERROR: {error}"
            )

        else:

            print(
                f"       prediction="
                f"{predicted_label} | "
                f"status={parse_status} | "
                f"correct={correct}"
            )

        print(
            f"       Saved: "
            f"{len(records_by_key)}/"
            f"{total_tasks}"
        )

    # -----------------------------------------------------------------
    # Final verification
    # -----------------------------------------------------------------

    print()

    print("=" * 70)
    print("PILOT COMPLETE")
    print("=" * 70)

    print(
        f"Expected requests:       {total_tasks}"
    )

    print(
        f"Saved unique requests:   "
        f"{len(records_by_key)}"
    )

    unique_keys = set(
        records_by_key.keys()
    )

    print(
        f"Unique request keys:     "
        f"{len(unique_keys)}"
    )

    valid_predictions = sum(
        1
        for record in records_by_key.values()
        if record.get("predicted_label")
        in VALID_LABELS
    )

    parse_failures = sum(
        1
        for record in records_by_key.values()
        if record.get("parse_status")
        == "unparseable"
    )

    api_errors = sum(
        1
        for record in records_by_key.values()
        if record.get("parse_status")
        == "api_error"
    )

    correct_predictions = sum(
        1
        for record in records_by_key.values()
        if record.get("correct") is True
    )

    print(
        f"Valid predictions:       "
        f"{valid_predictions}"
    )

    print(
        f"Unparseable responses:   "
        f"{parse_failures}"
    )

    print(
        f"API errors:              "
        f"{api_errors}"
    )

    print(
        f"Correct predictions:     "
        f"{correct_predictions}"
    )

    if valid_predictions > 0:

        accuracy = (
            correct_predictions
            / valid_predictions
        )

        print(
            f"Pilot accuracy*:         "
            f"{accuracy:.4f}"
        )

        print(
            "* Descriptive pilot statistic only; "
            "not final research evidence."
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()

    if len(records_by_key) == total_tasks:

        print(
            "All 160 unique pilot requests "
            "are present."
        )

    else:

        print(
            "WARNING: The expected number of "
            "requests was not saved."
        )

    print(
        "=" * 70
    )

    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the 20-sentence Financial "
            "PhraseBank pilot."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the 160-request "
            "experiment without API calls."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------

    if not PILOT_PATH.exists():

        raise FileNotFoundError(
            f"Pilot dataset not found:\n"
            f"{PILOT_PATH}"
        )

    if not FEW_SHOT_PATH.exists():

        raise FileNotFoundError(
            f"Few-shot dataset not found:\n"
            f"{FEW_SHOT_PATH}"
        )

    # -------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------

    pilot_df = pd.read_csv(
        PILOT_PATH,
        encoding="utf-8-sig",
    )

    few_shot_df = pd.read_csv(
        FEW_SHOT_PATH,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Validate pilot dataset
    # -------------------------------------------------------------

    required_columns = {
        "id",
        "sentence",
        "label",
    }

    missing_columns = (
        required_columns
        - set(pilot_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Pilot dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(pilot_df) != 20:

        raise ValueError(
            f"Expected 20 pilot rows, "
            f"found {len(pilot_df)}."
        )

    if pilot_df["id"].nunique() != 20:

        raise ValueError(
            "Pilot dataset contains "
            "duplicate IDs."
        )

    # -------------------------------------------------------------
    # Check few-shot leakage
    # -------------------------------------------------------------

    pilot_ids = set(
        pilot_df["id"]
    )

    few_shot_ids = set(
        few_shot_df["id"]
    )

    overlap = (
        pilot_ids
        & few_shot_ids
    )

    if overlap:

        raise ValueError(
            "Pilot/few-shot ID overlap detected: "
            f"{sorted(overlap)}"
        )

    # -------------------------------------------------------------
    # Build tasks
    # -------------------------------------------------------------

    tasks = build_tasks(
        pilot_df
    )

    # -------------------------------------------------------------
    # Dry run
    # -------------------------------------------------------------

    if args.dry_run:

        run_dry_run(
            tasks
        )

        return

    # -------------------------------------------------------------
    # API clients
    # -------------------------------------------------------------

    clients = create_clients()

    print()

    print(
        "Available API clients:"
    )

    for model in MODELS:

        if model in clients:

            print(
                f"  {model}: READY"
            )

        else:

            print(
                f"  {model}: NOT AVAILABLE"
            )

    missing_clients = [
        model
        for model in MODELS
        if model not in clients
    ]

    if missing_clients:

        raise RuntimeError(
            "Missing API clients for: "
            + ", ".join(missing_clients)
        )

    # -------------------------------------------------------------
    # Run pilot
    # -------------------------------------------------------------

    run_pilot(
        tasks=tasks,
        few_shot=few_shot_df,
        clients=clients,
    )


if __name__ == "__main__":
    main()
    