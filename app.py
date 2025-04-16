import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set page config as the very first Streamlit command
st.set_page_config(page_title="HoaxBuster", page_icon="📰", layout="wide")

# Inisialisasi NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing teks
def clean(text):
    text = str(text).lower()
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    punct = set(string.punctuation)
    text = "".join([ch for ch in text if ch not in punct])
    return text

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(text):
    word_tokens_no_stopwords = [w for w in text if w not in stop_words]
    return word_tokens_no_stopwords

def preprocess(text):
    text = clean(text)
    text = tokenize(text)
    text = remove_stop_words(text)
    return text

# Cache model dan tokenizer untuk efisiensi
@st.cache_resource
def load_lstm_model():
    return load_model('hoax_lstm_model.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

# Muat model dan tokenizer
model = load_lstm_model()
tokenizer = load_tokenizer()

# Parameter tokenisasi
max_features = 5000
max_len = 300  # Sesuaikan dengan pelatihan

# UI Streamlit
st.title("📰 HoaxBuster: Deteksi Berita Hoax")
st.markdown("""
    Masukkan teks berita di bawah ini untuk memeriksa apakah berita tersebut **hoax** atau **valid**.
    Aplikasi ini menggunakan model LSTM untuk analisis teks berbahasa Indonesia.
""")

# Input teks
news_text = st.text_area("Teks Berita", placeholder="Tempel teks berita di sini...", height=200)

# Tombol prediksi
if st.button("🔍 Periksa Berita", type="primary"):
    if news_text.strip() == "":
        st.warning("Mohon masukkan teks berita!", icon="⚠️")
    else:
        with st.spinner("Menganalisis teks..."):
            # Preprocessing teks
            processed_text = preprocess(news_text)
            st.write("Teks setelah preprocessing:", processed_text)
            text_seq = tokenizer.texts_to_sequences([" ".join(processed_text)])
            st.write("Urutan token:", text_seq)
            if not text_seq[0]:
                st.warning("Teks tidak dapat diproses. Pastikan teks berisi kata-kata yang relevan.")
                st.stop()
            text_padded = pad_sequences(sequences=text_seq, maxlen=max_len, padding='pre')
            
            # Prediksi
            prediction = model.predict(text_padded)
            st.write("Probabilitas prediksi (valid, hoax):", prediction[0])
            pred_class = np.argmax(prediction, axis=1)[0]
            pred_prob = prediction[0][pred_class] * 100

            # Tampilkan hasil dengan threshold
            threshold = 0.6  # Sesuaikan threshold untuk mengurangi bias ke hoax
            hoax_prob = prediction[0][1]
            pred_class = 1 if hoax_prob > threshold else 0
            pred_prob = hoax_prob * 100 if pred_class == 1 else (1 - hoax_prob) * 100

            if pred_class == 1:
                st.error(f"**Peringatan**: Berita ini kemungkinan **HOAX** (Kepercayaan: {pred_prob:.2f}%)", icon="🚨")
            else:
                st.success(f"**Hasil**: Berita ini kemungkinan **VALID** (Kepercayaan: {pred_prob:.2f}%)", icon="✅")

# Footer
st.markdown("---")
st.markdown("© 2025 HoaxBuster. Dibuat untuk mendeteksi berita hoax dengan AI.")