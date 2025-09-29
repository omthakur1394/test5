import json
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
MODEL_PATH = "mymodel.h5"
WORD_INDEX_PATH = "word_index.json"
MAX_LEN = 500

# --- APP INITIALIZATION ---
app = FastAPI(title="Sentiment Analysis API")

# --- ENABLE CORS ---
# For development, allow all. For production, replace "*" with your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://your-frontend.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL AND WORD INDEX AT STARTUP ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(WORD_INDEX_PATH, "r") as f:
    word_index = json.load(f)

# --- HELPER FUNCTION ---
def preprocess_text(text: str):
    """Convert input text into padded sequence for the model."""
    words = text.lower().split()
    sequence_of_indices = [word_index.get(word, 2) for word in words]  # 2 = <oov>
    padded_sequence = pad_sequences([sequence_of_indices], maxlen=MAX_LEN)
    return padded_sequence

# --- DATA MODEL ---
class ReviewRequest(BaseModel):
    text: str

# --- API ENDPOINTS ---
@app.post("/predict")
def predict_sentiment(review: ReviewRequest):
    """Predict sentiment from text review."""
    preprocessed_input = preprocess_text(review.text)
    prediction = model.predict(preprocessed_input)
    score = float(prediction[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"
    # CORRECTED CONFIDENCE CALCULATION
    confidence = score if score > 0.5 else 1 - score

    return {"sentiment": sentiment, "confidence": confidence}

@app.get("/")
def read_root():
    """Root endpoint to confirm API is running."""
    return {"message": "Sentiment Analysis API is running!"}