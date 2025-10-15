# src/preprocess.py
import os
import re
import string
import json
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
# Imports already use tensorflow.keras, which is correct
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text, stop_words):
    if pd.isna(text):
        return "placeholder"
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+', '', text)
    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w.lower() not in stop_words]
    cleaned = " ".join(tokens)
    return cleaned if cleaned else "placeholder"

def main(true_csv, fake_csv, out_csv, tokenizer_path, max_features=10000):
    stop_words = set(stopwords.words('english'))
    true = pd.read_csv(true_csv)
    fake = pd.read_csv(fake_csv)
    true['category'] = 1
    fake['category'] = 0
    df = pd.concat([true, fake]).reset_index(drop=True)
    df['clean_text'] = df['text'].apply(lambda t: clean_text(t, stop_words))
    df.to_csv(out_csv, index=False)
    # create tokenizer on clean_text
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df['clean_text'])
    # save tokenizer to json
    tok_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tok_json)
    print("Saved cleaned data to", out_csv)
    print("Saved tokenizer to", tokenizer_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", default="data/True.csv")
    parser.add_argument("--fake", default="data/Fake.csv")
    parser.add_argument("--out", default="data/cleaned.csv")
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    args = parser.parse_args()
    main(args.true, args.fake, args.out, args.tokenizer)
