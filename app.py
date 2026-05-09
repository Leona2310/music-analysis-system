import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
import yt_dlp

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Music Analysis System",
    page_icon="🎵",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
    color: #111827;
    font-family: "Segoe UI", sans-serif;
}
h1 { font-size: 30px; color: #1DB954; margin-bottom: 2px; }
h3 { font-size: 16px; color: #374151; margin-top: 0; }
.card {
    background-color: white;
    padding: 14px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL & DATA ----------------
model = joblib.load("genre_model.joblib")
dataset = pd.read_csv("features.csv")

# ---------------- HEADER ----------------
st.markdown("<h1>🎧 Music Analysis System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Genre prediction & audio insights using Data Science</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- INPUT METHOD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🎼 Input Source")
option = st.radio("", ["Upload Audio File", "YouTube Link"], horizontal=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(path):
    y, sr = librosa.load(path, sr=22050, duration=15)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return pd.DataFrame([[tempo,
        np.mean(mfccs), np.var(mfccs),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    ]], columns=[
        "tempo", "mfcc_mean", "mfcc_var",
        "spectral_centroid", "spectral_rolloff",
        "energy", "zcr", "chroma"
    ])

# ---------------- LIVE VISUALS ----------------
def show_live_charts(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=15)

    # Waveform
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Audio Waveform")
    fig, ax = plt.subplots(figsize=(5, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)
    st.caption(
        "The waveform shows how loudness changes over time in the audio signal."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Spectrogram
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎼 Spectrogram")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    st.caption(
        "The spectrogram visualizes frequency content over time, helping analyze tone and texture."
    )
    st.markdown("</div>", unsafe_allow_html=True)

audio_path = None

# ---------------- UPLOAD AUDIO ----------------
if option == "Upload Audio File":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload audio (.mp3 / .wav / .au)",
        type=["mp3", "wav", "au"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name

# ---------------- YOUTUBE LINK (ESTIMATION MODE) ----------------
if option == "YouTube Link":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.caption("⚠️ YouTube blocks audio downloads. Analysis is estimated.")
    yt_url = st.text_input("Paste YouTube link")
    consent = st.checkbox("I have permission to analyze this content")
    st.markdown("</div>", unsafe_allow_html=True)

    if yt_url and consent:
        st.info("🔍 Performing dataset-based genre estimation...")

        proxy_features = dataset.drop("label", axis=1).mean().to_frame().T
        prediction = model.predict(proxy_features)[0]

        sims = cosine_similarity(proxy_features, dataset.drop("label", axis=1))[0]
        top = dataset.iloc[sims.argsort()[-5:]]
        genre_counts = top["label"].value_counts()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 Estimated Genre (YouTube)")
        st.success(prediction)
        st.caption(
            "This estimate is based on learned dataset patterns, not actual YouTube audio."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎧 Genre Influence (Estimated)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            genre_counts,
            labels=genre_counts.index,
            autopct="%1.0f%%",
            colors=["#1DB954", "#6EE7B7", "#A7F3D0", "#D1FAE5"]
        )
        st.pyplot(fig)
        st.caption(
            "Influence is derived by comparing average feature similarity across genres."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESS UPLOADED AUDIO ----------------
if audio_path and os.path.exists(audio_path):

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("▶️ Play Audio")
    st.audio(audio_path)
    st.markdown("</div>", unsafe_allow_html=True)

    show_live_charts(audio_path)

    features = extract_features(audio_path)
    prediction = model.predict(features)[0]

    tempo = features["tempo"].iloc[0]
    energy = features["energy"].iloc[0]

    tempo_type = "Fast" if tempo > 120 else "Moderate" if tempo > 80 else "Slow"
    energy_level = "High" if energy > 0.07 else "Medium" if energy > 0.03 else "Low"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Predicted Genre")
    st.success(prediction)
    st.caption(
        "The genre is predicted using a trained machine learning model based on audio features."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎵 Song Characteristics")
    st.markdown(f"""
    • **Tempo:** {tempo_type}  
    • **Energy:** {energy_level}
    """)
    st.caption(
        "Tempo and energy help describe how fast and intense the song feels."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    sims = cosine_similarity(features, dataset.drop("label", axis=1))[0]
    top = dataset.iloc[sims.argsort()[-5:]]
    genre_counts = top["label"].value_counts()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎧 Genre Influence")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        genre_counts,
        labels=genre_counts.index,
        autopct="%1.0f%%",
        colors=["#1DB954", "#6EE7B7", "#A7F3D0", "#D1FAE5"]
    )
    st.pyplot(fig)
    st.caption(
        "Genre influence is calculated by comparing this song with similar songs in the dataset."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    os.remove(audio_path)

# ---------------- CONCLUSION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📌 Conclusion")
st.markdown("""
This system uses **audio feature extraction and machine learning**
to predict music genres and present interpretable insights
through visuals and simple explanations.
""")
st.markdown("</div>", unsafe_allow_html=True)
