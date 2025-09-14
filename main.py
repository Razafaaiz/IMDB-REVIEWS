# main.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load model
model = load_model("simple_rnn_imdb.h5")

# Helper functions
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="wide")

# Custom CSS for clean UI
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
            color: #222222;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        }
        textarea {
            border-radius: 8px !important;
        }
        .stButton button {
            background: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: #1d4ed8;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
An AI-powered NLP project that classifies movie reviews as Positive or Negative using an RNN trained on the IMDB dataset.

✨ Highlights:

Real-time sentiment prediction with confidence scores

Clean, professional Streamlit UI

Text preprocessing (tokenization & padding)

Portfolio-ready project showcasing Deep Learning + Deployment

🛠 Tech Stack: Python (TensorFlow, NumPy), RNN, Streamlit

📌 Applications: Movie reviews, customer feedback, social media sentiment analysis
""")

# Main content
#st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review and let AI classify it as **Positive** or **Negative**.")

# Input
user_input = st.text_area("✍️ Write your review here:", "")

if st.button("🔍 Classify"):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]
        sentiment = "Positive 😃" if score > 0.5 else "Negative 😞"

        st.subheader(f"**Result: {sentiment}**")
        st.progress(int(score * 100))
        st.write(f"**Confidence Score:** {score:.2f}")
    else:
        st.warning("⚠️ Please enter a review before classifying.")
st.markdown("</div>", unsafe_allow_html=True)


