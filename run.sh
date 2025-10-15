#!/bin/bash

echo ">>> Starting pipeline..."

echo ">>> Step 1: Cleaning & preprocessing data"
python src/preprocess.py --true data/True.csv --fake data/Fake.csv --out data/cleaned.csv --tokenizer models/tokenizer.json

echo ">>> Step 2: Training model"
python src/train.py --clean data/cleaned.csv --tokenizer models/tokenizer.json --glove glove.twitter.27B.100d.txt --out models/model.h5 --epochs 3

echo ">>> Step 3: Testing model"
python src/predict.py --model models/model.h5 --tokenizer models/tokenizer.json --text "This is a test article about the economy."

echo ">>> All done 🎉"
