import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 

@st.cache_resource
def load_artifacts():
    model = load_model('simple_rnn_imdb.h5', compile=False)
    return model

model = load_artifacts()  

st.title("Review's Sentiment Analysis")
st.write("Enter your review, and see the Sentiment!")

description = st.text_area("Enter a short description:", placeholder="Type your movie review here...")

def pre_processtext(text, word_index, maxlen=500):
    words = text.lower().split()
    # Map words -> imdb ids (shifted by +3 to account for special tokens)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

def predict_sentiment(review, model, word_index, threshold=0.5):
    pre_processed_input = pre_processtext(review, word_index)
    prediction = model.predict(pre_processed_input, verbose=0)
    prob = float(prediction[0][0])
    sentiment = 'Positive' if prob >= threshold else 'Negative'
    return sentiment, prob

if description and st.button("Predict Sentiment"):
    try:
        word_index = imdb.get_word_index()
        sentiment, prob = predict_sentiment(description, model, word_index)

        st.subheader(f"Prediction: **{sentiment}**")
        st.write(f"Probability (positive class): **{prob:.3f}**")
        st.progress(min(max(prob, 0.0), 1.0))
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

with st.expander("How this works"):
    st.write("""
    This app predicts the **sentiment** (positive or negative) of a movie review using a
    RNN trained on the **IMDB movie reviews dataset**.

    This means the app isn't just matching keywords, it’s learned patterns of language that often appear in positive or negative reviews.
    """)