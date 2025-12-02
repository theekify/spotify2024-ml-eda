import streamlit as st
import pickle
import numpy as np

# Load model
with open("C:\\Projects\\Languages\\AI_ML\\week1\\day4\\spotify_stream_predictor.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎧 Spotify Stream Predictor (2024)")

st.write("Enter the song metadata to predict total Spotify streams")

# Inputs
playlist_count = st.number_input("Spotify Playlist Count", 0)
playlist_reach = st.number_input("Spotify Playlist Reach", 0)
track_score = st.number_input("Track Score", 0.0)
spotify_pop = st.number_input("Spotify Popularity (0–100)", 0)
yt_views = st.number_input("YouTube Views", 0)
apple_playlists = st.number_input("Apple Music Playlist Count", 0.0)
tiktok_views = st.number_input("TikTok Views", 0)

if st.button("Predict Streams"):
    data = np.array([[playlist_count, playlist_reach, track_score,
                      spotify_pop, yt_views, apple_playlists, tiktok_views]])
    
    prediction = model.predict(data)[0]
    st.success(f"Estimated Streams: {int(prediction):,}")
