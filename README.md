# Sentiment Prediction Using RNN

Binary sentiment analysis on the IMDB movie review dataset using a lightweight Simple Recurrent Neural Network (SimpleRNN) built with TensorFlow/Keras.

---

## Overview

- **Task:** Classify a movie review as *Positive* or *Negative*
- **Dataset:** IMDB movie reviews (25k train / 25k test), pre-tokenised
- **Architecture:** `Embedding -> SimpleRNN -> Dense (Sigmoid)`


## Input Handling

- Only the **top 10,000 most frequent words** are used  
- All other words are replaced with `<UNK>`  
- Each review is **padded/truncated to 500 tokens** to maintain a fixed input size


## Model Architecture

| Layer      | Description                                           |
|-----------|-------------------------------------------------------|
| Embedding | Maps 10,000-word vocabulary → 128-dim dense vectors    |
| SimpleRNN | 128 units, ReLU activation – processes sequence        |
| Dense     | 1 unit, Sigmoid – outputs probability of sentiment     |


## Training Setup

- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam (learning rate = 0.0001)  
- **Callback:** `EarlyStopping(patience=5, monitor='val_loss')`


## Evaluation Metrics

- Accuracy  
- Precision / Recall / F1-Score  
- ROC-AUC  
- Confusion Matrix (TP, FP, TN, FN)

**Performance:**  
~88% accuracy and ~0.95 ROC-AUC on test set.


## Single Review Prediction Pipeline

1. Raw review text  
2. Tokenise using IMDB vocabulary  
3. Unknown words -> `<UNK>`  
4. Pad/truncate to 500 tokens  
5. Pass into the trained model  
6. Output **Positive** / **Negative** sentiment probability

**Example**

| Input Text                         | Prediction | Sentiment |
|------------------------------------|------------|-----------|
| `"This movie was surprisingly good!"` | 0.91       | Positive  |

## Possible Extensions

- Replace SimpleRNN with **LSTM** or **GRU** for better context modelling  
- Use **pre-trained word embeddings** (e.g., GloVe)  
- Add **Dropout** or **Batch Normalization** for regularisation

## Applications of RNNs

Recurrent Neural Networks are designed to handle **sequential data**, where the order and context of information matters. Unlike traditional feed-forward networks like ANNs, RNNs have a memory mechanism that allows them to model **temporal dependencies**, making them highly suited for tasks involving time-series or language.

**Key strengths of RNNs:**
- Capture patterns across sequences over time (temporal trends, behaviour progression)
- Handle variable-length inputs (e.g. sentences, log sequences, events)
- Learn from contextual relationships rather than individual data points

## Use-Cases of RNNs in the Insurance Domain

- **Claims Fraud Detection:**  
  Model sequences of claim events and flag unusual temporal patterns in how a claim is reported or progresses through different stages.

- **Customer Behaviour Sequence Modelling / Churn Prediction:**  
  Instead of static attributes, track sequences of customer actions (e.g logins, payments, enquiries) over time to detect behavioural drift signalling churn.

- **Time-Series Forecasting for Loss Prediction:**  
  Predict future claim amounts/frequency using historical payout sequences,

- **Automated Claims Processing (NLP):**  
  Use RNNs to read and understand free-text claim descriptions or adjuster notes in order to route, summarise, or prioritise claims.

- **Underwriting Risk Signal Modelling:**  
  Analyse sequences of sensor / telematics data (e.g driving behaviour in motor insurance or IoT devices in home insurance) to update risk scores.


## Streamlit App
https://sentiment-prediction-using-rnn-ns2h9ybqwbgcsnulb9baa4.streamlit.app
