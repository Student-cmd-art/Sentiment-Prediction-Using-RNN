# Sentiment-Prediction-Using-RNN

Binary sentiment analysis on IMDB movie reviews using a lightweight Simple Recurrent Neural Network (SimpleRNN) built with TensorFlow/Keras.

Overview
Task: Classify a movie review as Positive or Negative.
Dataset: Built-in IMDB dataset (25k train + 25k test), already tokenized.
Architecture: Embedding → SimpleRNN → Sigmoid.

Input Handling:
Only top 10,000 most frequent words are used (others become <UNK>).
Reviews are padded/truncated to 500 tokens for fixed input size.

Model Architecture
Layer	Description
Embedding	10,000-word vocab → 128-dimensional dense vectors
SimpleRNN	128 units, ReLU activation, processes tokens sequentially
Dense	1 unit, Sigmoid → outputs probability of positive sentiment

Training Setup
Loss: Binary Cross-Entropy
Optimizer: Adam (learning rate = 0.0001)
EarlyStopping: patience = 5 (monitors val_loss)

Evaluation Metrics
Accuracy
Precision / Recall / F1-Score
ROC-AUC
Confusion Matrix (TP, FP, TN, FN)
Typical performance is ~88% accuracy and 0.95 ROC-AUC on the hold-out test set.

Single Review Prediction
Raw text is:
Tokenised using the IMDB vocabulary
Unknown words → <UNK>
Padded to 500 words
Passed through the model to output a POSITIVE vs NEGATIVE probability
Example:
Input: "This movie was surprisingly good!"
Prediction: 0.91 → Positive

Possible Extensions
Swap SimpleRNN for LSTM/GRU to capture longer-term context
Use pretrained embeddings (e.g. GloVe)
Add dropout or batch normalization
