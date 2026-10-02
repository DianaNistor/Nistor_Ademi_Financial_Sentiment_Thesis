import pandas as pd


def direct_zero_shot(sentence: str) -> str:
    return f"""
Classify the sentiment of the following financial statement.

Choose exactly one label:
positive
neutral
negative

Statement:
{sentence}

Return only the label.
""".strip()


def direct_few_shot(
    sentence: str,
    examples: pd.DataFrame,
) -> str:
    examples_text = "\n\n".join(
        f"Statement: {row.sentence}\n"
        f"Label: {row.label}"
        for row in examples.itertuples()
    )

    return f"""
Classify the sentiment of the following financial statement.

Choose exactly one label:
positive
neutral
negative

Here are correctly classified examples:

{examples_text}

Now classify this statement:

Statement:
{sentence}

Return only the label.
""".strip()


def cot_zero_shot(sentence: str) -> str:
    return f"""
Analyze the sentiment of the following financial statement.

Think step by step about whether the information represents
a positive, neutral, or negative financial development.

Statement:
{sentence}

After your analysis, provide your final answer as:

FINAL: positive
FINAL: neutral
or
FINAL: negative
""".strip()


def cot_few_shot(
    sentence: str,
    examples: pd.DataFrame,
) -> str:
    examples_text = "\n\n".join(
        f"Statement: {row.sentence}\n"
        f"Label: {row.label}"
        for row in examples.itertuples()
    )

    return f"""
Analyze the sentiment of the following financial statement.

Choose one sentiment:
positive
neutral
negative

Here are correctly classified examples:

{examples_text}

Now analyze this statement:

Statement:
{sentence}

Think step by step about whether the information represents
a positive, neutral, or negative financial development.

After your analysis, provide your final answer as:

FINAL: positive
FINAL: neutral
or
FINAL: negative
""".strip()
