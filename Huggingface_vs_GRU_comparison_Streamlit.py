#Streamlit File

import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.nn.utils.rnn import pad_sequence
from transformers import pipeline

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === GRU Model Definition ===
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=128, num_layers=2, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Linear(hidden_size * 2, 2)  # Binary classification
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        h0 = torch.zeros(self.gru.num_layers * 2, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        attention_weights = self.attention(out)
        context_vector = torch.sum(attention_weights * out, dim=1)

        context_vector = self.norm(self.dropout(context_vector))
        logits = self.fc(context_vector)
        return logits

# === Load model and vocab ===
@st.cache_resource
def load_model_and_vocab():
    with open("gru_sentiment_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    word2idx = model_data['word2idx']
    config = model_data['model_config']

    model = GRUModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        pretrained_embeddings=None
    )
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    return model, word2idx

# === Load Hugging Face sentiment classifier ===
@st.cache_resource
def load_huggingface_pipeline():
    return pipeline("sentiment-analysis")

model, word2idx = load_model_and_vocab()
hf_classifier = load_huggingface_pipeline()

# === Preprocessing ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    indexed = [word2idx.get(word, 0) for word in tokens]
    tensor = torch.tensor(indexed, dtype=torch.long)
    padded = pad_sequence([tensor], batch_first=True)
    return padded

# === Streamlit UI ===
st.title("🧠 GRU vs 🤗 Transformer Sentiment Classifier")
review = st.text_area("Enter a review:", "")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # GRU model prediction
        input_tensor = preprocess_text(review)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

        st.subheader("🔮 GRU Model Prediction")
        st.success(f"**{sentiment}**")

        # Hugging Face transformer prediction
        hf_result = hf_classifier(review)[0]
        hf_label = hf_result['label']
        hf_score = hf_result['score']

        st.subheader("🤗 Hugging Face Transformer Prediction")
        st.info(f"**{hf_label}** ({hf_score:.2%} confidence)")
