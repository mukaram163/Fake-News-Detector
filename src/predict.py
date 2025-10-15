# src/predict.py
import argparse
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from src.preprocess import clean_text
from nltk.corpus import stopwords

def main(model_path, tokenizer_path, text_input, maxlen=300):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok = tokenizer_from_json(f.read())
        
    # ✅ No need for custom_objects now
    model = load_model(model_path, compile=False)
    
    stop_words = set(stopwords.words('english'))
    cleaned = clean_text(text_input, stop_words)
    seq = tok.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=maxlen)
    prob = model.predict(seq)[0][0]
    label = "Not Fake" if prob > 0.5 else "Fake"
    print(f"Probability Real: {prob:.4f} -> Prediction: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.keras")  # ✅ Updated from .h5
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    main(args.model, args.tokenizer, args.text)
