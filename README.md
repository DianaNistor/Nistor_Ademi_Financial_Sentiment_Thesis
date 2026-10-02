# Financial PhraseBank experiment dataset

This package prepares the dataset for the LLM financial-sentiment experiment.

## Design
- Source: Financial PhraseBank
- Configuration: `sentences_allagree` (100% annotator agreement)
- Expected corpus size: 2,264 sentences
- Labels: negative, neutral, positive
- Few-shot examples: 9 total (3 per label)
- Random seed: 42
- Evaluation set: all remaining 2,255 sentences
- The few-shot examples are excluded from the evaluation set to prevent leakage.

## Why the corpus itself is not bundled here
Financial PhraseBank is distributed under CC BY-NC-SA 3.0. The included script retrieves
the dataset from its published Hugging Face dataset repository and creates the exact CSVs
locally, preserving the source and licensing information.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
python prepare_dataset.py
```

It creates:
- `dataset/financial_phrasebank_allagree.csv`
- `dataset/few_shot_examples.csv`
- `dataset/test_dataset.csv`

Each CSV has:
- `id`
- `sentence`
- `label`

## Reproducibility
Do not change `SEED = 42` or `EXAMPLES_PER_LABEL = 3` after the experiment starts.
Keep the generated CSV files with the experiment results so every LLM receives the
same test items and the same few-shot examples.

## Important methodology note
Before running the LLM experiment, confirm with the supervisor that:
1. 100%-agreement PhraseBank is the intended subset.
2. Nine few-shot examples (3 per sentiment) are acceptable.
3. The precise Chain-of-Thought prompt design is approved.
