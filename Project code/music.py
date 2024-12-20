import streamlit as st
import cv2
import pygame
from deepface import DeepFace
import random
import time
import numpy as np
from PIL import Image

# Initialize pygame mixer for music playback
pygame.mixer.init()

# Music files mapped to emotions
music_map = {
    "angry": "Musics/Soft_and_Furious_-_05_-_Through_the_water_and_rain(chosic.com).mp3",
    "sad": "Musics/Rodrigo-Excerpt-From-Concierto-Aranquez-mosbat.net.mp3",
    "happy": "Musics/Uplifting_Summer_Tropical_Dance.mp3",
    "surprise": "Musics/MediaBaz.net-whoah-.mp3",
    "neutral": "Musics/meditation-music-blog-(6).mp3",
    "fear": "Musics/Download Remix Tarsnak Bikalam Nubmer 01 [Melodya].mp3"
}

# Start Video Feed
def start_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return None
    return cap

# Capture video feed and emotion recognition
def update_video_feed(cap):
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    return None

# Take a photo and recognize emotion
def take_photo(cap):
    ret, frame = cap.read()
    if ret:
        photo_taken = frame
        try:
            result = DeepFace.analyze(photo_taken, actions=["emotion"], enforce_detection=False)
            dominant_emotion = result[0]["dominant_emotion"]
            confidence_score = result[0]["emotion"][dominant_emotion]
            return dominant_emotion, confidence_score, photo_taken
        except Exception as e:
            st.error(f"Error during emotion detection: {e}")
            return None, None, None
    return None, None, None

# Play music based on detected emotion
def play_music_based_on_emotion(emotion):
    if emotion in music_map:
        music_file = music_map[emotion]
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play(-1)  # Loop the music
        except pygame.error:
            st.error(f"Error: Could not play {music_file}. Ensure the file exists and is supported.")
    else:
        st.warning(f"No music mapped for emotion: {emotion}")

# Stop the music
def stop_music():
    pygame.mixer.music.stop()

# Stop the video feed
def stop_video(cap):
    if cap is not None:
        cap.release()

# Streamlit UI
st.title("Real-Time Emotion Detection and Music Player")

# Initialize webcam and emotion recognition
cap = start_video()

if cap:
    # Video feed display
    stframe = st.empty()

    # Emotion label
    emotion_label = st.empty()

    # Buttons for control
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        start_button = st.button("Start Video", on_click=start_video)

    with col2:
        stop_button = st.button("Stop Video", on_click=stop_video, args=(cap,))

    with col3:
        take_photo_button = st.button("Take Photo")
        
    # Start Video Feed
    if start_button:
        stframe.image(update_video_feed(cap), channels="RGB", use_column_width=True)
    
    # Take Photo and detect emotion
    if take_photo_button:
        dominant_emotion, confidence_score, photo_taken = take_photo(cap)
        if dominant_emotion:
            emotion_label.text(f"Emotion: {dominant_emotion} (Confidence: {confidence_score:.2f})")
            play_music_based_on_emotion(dominant_emotion)
    
    # Stop music
    if stop_button:
        stop_music()
        emotion_label.text("Music stopped.")

    # Optionally stop video feed
    if stop_button:
        stop_video(cap)

    # Release camera after the session ends
    cap.release()

