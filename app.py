import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences  # ✅ FIXED

# --- CONFIGURATION ---
MODEL_PATH = "mymodel.h5"
WORD_INDEX_PATH = "word_index.json"
MAX_LEN = 500

# --- APP INITIALIZATION ---
app = FastAPI(title="Sentiment Analysis API")

# --- LOAD MODEL AND WORD INDEX AT STARTUP ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(WORD_INDEX_PATH) as f:
    word_index = json.load(f)

# --- HELPER FUNCTION ---
def preprocess_text(text: str):
    """Preprocesses text for the model."""
    words = text.lower().split()
    sequence_of_indices = [word_index.get(word, 2) for word in words]  # 2 is for <oov>
    padded_sequence = pad_sequences([sequence_of_indices], maxlen=MAX_LEN)  # ✅ FIXED
    return padded_sequence

# --- DATA MODELS ---
class ReviewRequest(BaseModel):
    text: str

# --- API ENDPOINT ---
@app.post("/predict")
def predict_sentiment(review: ReviewRequest):
    """Predicts sentiment from a text review."""
    preprocessed_input = preprocess_text(review.text)
    prediction = model.predict(preprocessed_input)
    score = float(prediction[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"
    confidence = score if sentiment == "Positive" else 1 - score

    return {"sentiment": sentiment, "confidence": confidence}

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Sentiment Analysis API is running!"}
