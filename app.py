import streamlit as st
import numpy as np
import pickle

# Load model using standalone keras (for .h5 compatibility)

from keras.models import load_model

# Use tensorflow utilities for preprocessing

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model

model = load_model("model/language_identification_model.h5", compile=False)

# Load tokenizer

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load labels

with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)

def predict_language(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=150)
    pred = model.predict(pad)
    return labels[np.argmax(pred)]

st.title("🌍 Language Detection App")

user_input = st.text_area("Enter Text")
if st.button("Detect Language"):
    if user_input:
        result = predict_language(user_input)
        st.success(f"Detected Language: {result}")
