import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from tensorflow.keras.preprocessing import sequence

# --- CONFIGURATION ---
MODEL_PATH = "mymodel.h5"
WORD_INDEX_PATH = "word_index.json"
MAX_LEN = 500

# --- APP INITIALIZATION ---
app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# --- LOAD MODEL AND WORD INDEX AT STARTUP ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(WORD_INDEX_PATH) as f:
    word_index = json.load(f)

# --- HELPER FUNCTION ---
def preprocess_text(text: str):
    """Preprocesses text for the model."""
    words = text.lower().split()
    sequence_of_indices = [word_index.get(word, 2) for word in words] # 2 is for <oov>
    padded_sequence = sequence.pad_sequences([sequence_of_indices], maxlen=MAX_LEN)
    return padded_sequence

# --- DATA MODELS ---
class ReviewRequest(BaseModel):
    text: str

# --- API ENDPOINTS ---
@app.get("/")
def read_root(request: Request):
    """Serves the initial HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_sentiment(review: ReviewRequest):
    """Predicts sentiment from a text review."""
    preprocessed_input = preprocess_text(review.text)
    prediction = model.predict(preprocessed_input)
    score = float(prediction[0][0]) # Ensure score is a standard float

    sentiment = "Positive" if score > 0.5 else "Negative"
    confidence = score if sentiment == "Positive" else 1 - score

    return {"sentiment": sentiment, "confidence": confidence}