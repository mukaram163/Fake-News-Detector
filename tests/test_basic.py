# tests/test_basic.py
from src.preprocess import clean_text
from nltk.corpus import stopwords

def test_clean_text_removes_punctuation():
    stop = set(stopwords.words('english'))
    text = "Hello!!! This, right here?? is a test."
    cleaned = clean_text(text, stop)
    assert "!" not in cleaned
    assert "," not in cleaned
    assert "?" not in cleaned

def test_clean_text_handles_empty():
    stop = set(stopwords.words('english'))
    cleaned = clean_text("", stop)
    assert cleaned == "placeholder"