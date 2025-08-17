import pytest
import sys
import os

# Add the src directory to the system path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_text

def test_clean_text_basic():
    """Test that the function correctly handles basic cleaning tasks."""
    raw_text = "I am writing to file a complaint about the company's service!"
    expected_clean_text = "writing file complaint companys service"
    assert clean_text(raw_text) == expected_clean_text

def test_clean_text_with_numbers():
    """Test that numbers and special characters are removed."""
    raw_text = "Complaint submitted on 10/12/2024. Phone number is 123-456-7890."
    expected_clean_text = "complaint submitted phone number"
    assert clean_text(raw_text) == expected_clean_text

def test_clean_text_empty_input():
    """Test that the function returns an empty string for non-string input."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text(123) == ""