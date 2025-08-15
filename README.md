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
