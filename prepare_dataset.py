from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
EXAMPLES_PER_LABEL = 3

RAW_FILE = Path("raw_data/Sentences_AllAgree.txt")
OUTPUT_DIR = Path("dataset")

EXPECTED_LABELS = {
    "positive",
    "neutral",
    "negative",
}


# --------------------------------------------------
# Load the original Financial PhraseBank dataset
# --------------------------------------------------

def load_dataset() -> pd.DataFrame:
    rows = []

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the dataset file:\n"
            f"{RAW_FILE.resolve()}\n\n"
            f"Make sure Sentences_AllAgree.txt is inside "
            f"the raw_data folder."
        )

    with open(RAW_FILE, "r", encoding="iso-8859-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            # Financial PhraseBank uses @ to separate
            # the sentence from the sentiment label.
            try:
                sentence, label = line.rsplit("@", 1)
            except ValueError:
                raise ValueError(
                    f"Could not parse line {line_number}:\n{line}"
                )

            rows.append(
                {
                    "sentence": sentence.strip(),
                    "label": label.strip().lower(),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Validate dataset
# --------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    print("\nChecking dataset...")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["sentence"].isna().any():
        raise ValueError("Some sentences are missing.")

    if df["label"].isna().any():
        raise ValueError("Some labels are missing.")

    actual_labels = set(df["label"].unique())

    if actual_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected labels found: {actual_labels}\n"
            f"Expected: {EXPECTED_LABELS}"
        )

    print("Dataset validation passed!")


# --------------------------------------------------
# Select few-shot examples
# --------------------------------------------------

def create_few_shot_examples(
    df: pd.DataFrame,
) -> pd.DataFrame:
    few_shot = (
        df.groupby("label", group_keys=False)
        .sample(
            n=EXAMPLES_PER_LABEL,
            random_state=SEED,
        )
        .sort_values(["label", "id"])
        .reset_index(drop=True)
    )

    return few_shot


# --------------------------------------------------
# Create test dataset
# --------------------------------------------------

def create_test_dataset(
    df: pd.DataFrame,
    few_shot: pd.DataFrame,
) -> pd.DataFrame:
    # Remove all few-shot examples from the test set.
    # This prevents data leakage.
    test = df[
        ~df["id"].isin(few_shot["id"])
    ].copy()

    return test.reset_index(drop=True)


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

def save_datasets(
    full_dataset: pd.DataFrame,
    few_shot: pd.DataFrame,
    test_dataset: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    full_dataset.to_csv(
        OUTPUT_DIR / "financial_phrasebank_allagree.csv",
        index=False,
    )

    few_shot.to_csv(
        OUTPUT_DIR / "few_shot_examples.csv",
        index=False,
    )

    test_dataset.to_csv(
        OUTPUT_DIR / "test_dataset.csv",
        index=False,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 60)
    print("Financial PhraseBank Dataset Preparation")
    print("=" * 60)

    print("\nLoading dataset from:")
    print(RAW_FILE.resolve())

    # Load
    df = load_dataset()

    print(f"\nDataset loaded successfully!")
    print(f"Total sentences: {len(df)}")

    # Add a permanent ID to every sentence
    df.insert(
        0,
        "id",
        [
            f"fpb_{i:04d}"
            for i in range(1, len(df) + 1)
        ],
    )

    # Validate
    validate_dataset(df)

    # Show label distribution
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Create few-shot examples
    few_shot = create_few_shot_examples(df)

    # Create test dataset
    test_dataset = create_test_dataset(
        df,
        few_shot,
    )

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------

    print("\nRunning safety checks...")

    assert len(few_shot) == 9, (
        f"Expected 9 few-shot examples, "
        f"but got {len(few_shot)}"
    )

    assert len(test_dataset) == len(df) - 9, (
        "Test dataset size is incorrect."
    )

    # Make sure none of the few-shot examples
    # accidentally appear in the test dataset.
    few_shot_ids = set(few_shot["id"])
    test_ids = set(test_dataset["id"])

    assert few_shot_ids.isdisjoint(test_ids), (
        "Data leakage detected! "
        "Few-shot examples exist in the test dataset."
    )

    # Make sure we have exactly 3 examples per label
    few_shot_counts = few_shot["label"].value_counts()

    for label in EXPECTED_LABELS:
        assert few_shot_counts[label] == EXAMPLES_PER_LABEL

    print("All safety checks passed!")

    # Save files
    save_datasets(
        full_dataset=df,
        few_shot=few_shot,
        test_dataset=test_dataset,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FEW-SHOT EXAMPLES")
    print("=" * 60)

    print(
        few_shot[
            ["id", "label", "sentence"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"Full dataset:       {len(df)}")
    print(f"Few-shot examples:  {len(few_shot)}")
    print(f"Test dataset:       {len(test_dataset)}")

    print("\nFew-shot distribution:")
    print(few_shot["label"].value_counts())

    print("\nFiles created:")

    print(
        "  dataset/"
        "financial_phrasebank_allagree.csv"
    )

    print(
        "  dataset/"
        "few_shot_examples.csv"
    )

    print(
        "  dataset/"
        "test_dataset.csv"
    )

    print("\nDataset preparation successful!")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
    