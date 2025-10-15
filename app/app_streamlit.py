# app/app_streamlit.py

# --- FIX: ADD PROJECT ROOT TO PATH ---
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# --------------------------------------

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from src.preprocess import clean_text
from nltk.corpus import stopwords


# --- CONFIGURATION ---
MODEL_PATH = "models/model.keras"   # ✅ Updated from .h5 to .keras
TOKENIZER_PATH = "models/tokenizer.json"
MAXLEN = 300
STOP_WORDS = set(stopwords.words('english'))


@st.cache_data
def load_resources():
    """Loads the model and tokenizer only once."""
    try:
        model = load_model(MODEL_PATH, compile=False)  # ✅ No custom_objects needed for .keras model
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tok = tokenizer_from_json(f.read())
        return model, tok
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        st.stop()
        

def predict_news(text, model, tokenizer, maxlen, stop_words):
    """Cleans text, converts to sequence, pads, and predicts probability."""
    if not text.strip():
        return 0.0
        
    cleaned = clean_text(text, stop_words)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=maxlen)
    
    prob = model.predict(seq, verbose=0)
    return float(prob[0][0])


# ====================================================================
# --- STREAMLIT UI CODE STARTS HERE ---
# ====================================================================

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("📰 AI-Powered Fake News Detector")
st.markdown("---")

model, tok = load_resources()

st.subheader("Enter the News Article Text")
text_input = st.text_area(
    "Paste the full news article text below:", 
    placeholder="Example: 'A new study reveals that eating chocolate cures the common cold.'",
    height=250
)

if st.button("Analyze News", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing text with deep learning model...'):
            real_prob = predict_news(text_input, model, tok, MAXLEN, STOP_WORDS)
        
        prediction = "REAL" if real_prob >= 0.5 else "FAKE"
        fake_prob = 1.0 - real_prob
        
        st.markdown("---")
        st.subheader("Analysis Result")
        
        if prediction == "REAL":
            st.metric(
                label="VERDICT", 
                value="REAL NEWS ✅", 
                delta=f"{round(real_prob * 100, 2)}% Confidence",
                delta_color="normal"
            )
        else:
            st.metric(
                label="VERDICT", 
                value="FAKE NEWS ❌", 
                delta=f"{round(fake_prob * 100, 2)}% Confidence",
                delta_color="inverse"
            )

        st.markdown("### Confidence Scores")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Real Probability:** `{round(real_prob * 100, 2)}%`")
            st.progress(real_prob, text="Probability of Real News")

        with col2:
            st.markdown(f"**Fake Probability:** `{round(fake_prob * 100, 2)}%`")
            st.progress(fake_prob, text="Probability of Fake News")

st.sidebar.markdown(
    """
    ## About the Detector
    This is a **Deep Learning Fake News Classifier** built with Keras/TensorFlow, 
    deployed using Docker and Streamlit.

    - **Model Type:** LSTM/RNN
    - **Base Image:** Python 3.10-slim
    - **Deployment:** Docker Container on port 8501
    """
)