import streamlit as st
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

# --- CONFIGURATION ---
MODEL_PATH = "mymodel.h5"
WORD_INDEX_PATH = "word_index.json"
MAX_LEN = 500

# --- HELPER FUNCTIONS ---

# Use st.cache_resource to load model and word_index only once
@st.cache_resource
def load_model_and_word_index():
    """Loads the saved Keras model and the word index."""
    print("Loading model and word index...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(WORD_INDEX_PATH, 'r') as f:
        word_index = json.load(f)
    print("Loading complete.")
    return model, word_index

def preprocess_text(text, word_index, max_len):
    """Preprocesses a single text string for prediction."""
    words = text.lower().split()
    sequence_of_indices = [word_index.get(word, 2) for word in words] # 2 is for <oov>
    padded_sequence = sequence.pad_sequences([sequence_of_indices], maxlen=max_len)
    return padded_sequence

# --- LOAD RESOURCES ---
model, word_index = load_model_and_word_index()


# --- STREAMLIT UI ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to find out if it's positive or negative!")

# User input text area
user_input = st.text_area("Movie Review", "This movie was absolutely fantastic! The acting was great and the plot was thrilling.")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input
        preprocessed_input = preprocess_text(user_input, word_index, MAX_LEN)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]

        # Display result
        st.subheader("Analysis Result")
        if score > 0.5:
            st.success(f"Positive Review 👍 (Confidence: {score:.2%})")
        else:
            st.error(f"Negative Review 👎 (Confidence: {1-score:.2%})")
    else:
        st.warning("Please enter a review to analyze.")