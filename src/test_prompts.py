from pathlib import Path

import pandas as pd

from prompts import (
    direct_zero_shot,
    direct_few_shot,
    cot_zero_shot,
    cot_few_shot,
)


DATASET_DIR = Path("dataset")

few_shot = pd.read_csv(
    DATASET_DIR / "few_shot_examples.csv"
)

test_dataset = pd.read_csv(
    DATASET_DIR / "test_dataset.csv"
)


# Take the first test observation
row = test_dataset.iloc[0]

sentence = row["sentence"]
correct_label = row["label"]


print("=" * 70)
print("TEST SENTENCE")
print("=" * 70)

print("Sentence:", sentence)
print("Correct label:", correct_label)


print("\n" + "=" * 70)
print("DIRECT ZERO-SHOT")
print("=" * 70)

print(direct_zero_shot(sentence))


print("\n" + "=" * 70)
print("DIRECT FEW-SHOT")
print("=" * 70)

print(direct_few_shot(sentence, few_shot))


print("\n" + "=" * 70)
print("COT ZERO-SHOT")
print("=" * 70)

print(cot_zero_shot(sentence))


print("\n" + "=" * 70)
print("COT FEW-SHOT")
print("=" * 70)

print(cot_few_shot(sentence, few_shot))
