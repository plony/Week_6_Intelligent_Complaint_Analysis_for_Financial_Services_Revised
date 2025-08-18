# tests/test_preprocessing.py

import pytest
import pandas as pd
from src.preprocessing import filter_and_clean_data, clean_text


@pytest.fixture
def sample_dataframe():
    data = {
        "Product": ["Credit card", "Savings account", "Personal loan"],
        "Consumer complaint narrative": [
            "This is a complaint about a credit card issue.",
            "I have an issue with my savings account.",
            "I am writing to file a complaint about my personal loan."
        ],
    }
    return pd.DataFrame(data)


def test_filter_and_clean_data(sample_dataframe):
    df = filter_and_clean_data(sample_dataframe)
    assert len(df) == 3
    assert "cleaned_narrative" in df.columns
    assert "i writing file complaint personal loan" in df[
        "cleaned_narrative"
    ].iloc[2]


def test_clean_text():
    text = "Hello, I am writing to file a complaint about this issue! 123"
    expected = "hello issue"
    assert clean_text(text) == expected