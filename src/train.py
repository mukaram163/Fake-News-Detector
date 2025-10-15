# src/train.py
import argparse
import numpy as np
import pandas as pd
# FIX: Change to tensorflow.keras imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import json
import os

# === MLflow Imports ===
import mlflow
# FIX: Change to mlflow.tensorflow for modern MLflow support
import mlflow.tensorflow 
# ======================

def load_embeddings(path, word_index, max_features, embed_size=100):
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(0, 1, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix

def build_model(max_features, embed_size, maxlen, embedding_matrix):
    model = Sequential()
    # FIX (Minor Keras Warning): Removed deprecated `input_length` argument 
    # (input_dim remains for vocabulary size)
    model.add(Embedding(input_dim=max_features, output_dim=embed_size,
                        weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(clean_csv, tokenizer_json, embedding_file, model_out, max_features=10000, maxlen=300, embed_size=100, batch_size=128, epochs=3):
    df = pd.read_csv(clean_csv)
    with open(tokenizer_json, 'r', encoding='utf-8') as f:
        tok = tokenizer_from_json(f.read())
    X = tok.texts_to_sequences(df['clean_text'])
    X = pad_sequences(X, maxlen=maxlen)
    y = df['category'].values
    # shuffle & split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.2)
    embedding_matrix = load_embeddings(embedding_file, tok.word_index, max_features, embed_size)
    model = build_model(max_features, embed_size, maxlen, embedding_matrix)
    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_out, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # === Start MLflow tracking ===
    mlflow.set_experiment("FakeNews-LSTM")

    with mlflow.start_run():
        # Log parameters (so you can see them in the MLflow dashboard)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("embedding_file", embedding_file)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Log final metrics
        # Check if history.history contains 'val_accuracy' (it should if using Keras/TensorFlow)
        if 'val_accuracy' in history.history:
            val_acc = history.history['val_accuracy'][-1]
            mlflow.log_metric("val_accuracy", val_acc)
            
        if 'val_loss' in history.history:
            val_loss = history.history['val_loss'][-1]
            mlflow.log_metric("val_loss", val_loss)
            
        # Log the model to MLflow
        mlflow.tensorflow.log_model(model, "model")

    print("Training complete. Best model saved to", model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="data/cleaned.csv")
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--glove", default="glove.twitter.27B.100d.txt")
    parser.add_argument("--out", default="models/model.h5")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args.clean, args.tokenizer, args.glove, args.out, epochs=args.epochs)
