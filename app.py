import streamlit as st
import requests
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import av
import math
import pickle
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')


engine.setProperty('voice', "english")
engine.setProperty('rate', 125)


import warnings
warnings.filterwarnings('ignore')
from squat_model.squat import SquatFormCorrectionTransformer
from bicep_model.bicep import BicepVideoTransformer
from lunge_model.lunge import LungeFormTransformer
from plank_model.plank import PlankPoseEstimator
st.set_page_config(page_title="FormFlow",page_icon="")

st.title("FormFlow")
st.markdown(
    """
    <style>
    select {
        cursor: pointer;
    }
    button {
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
# st.markdown("---")

exercise_col, animation_col = st.columns([2, 1])

# Exercise selection column
with exercise_col:
    st.write("## Select an Exercise")

    # List of exercise options
    exercise_options = ["Biceps","Squat","Plank","Lunge"]
    selected_exercise = st.selectbox("Choose an exercise:", exercise_options)
    
st.markdown("---")
exercise_transformers = {
    "Squat": SquatFormCorrectionTransformer,
    "Biceps": BicepVideoTransformer,
    "Lunge": LungeFormTransformer,
    "Plank": PlankPoseEstimator,
}



exercise_videos = {
    "Lunge": "https://www.youtube.com/watch?v=0pkjOk0EiAk&pp=ygUWcHVzaHVwc2V4Y2VyaXNlIHZpZGVvIA%3D%3D",
    "Squat": "https://www.youtube.com/watch?v=rMvwVtlqjTE&pp=ygUVc3F1YXQgZXhjZXJpc2UgdmlkZW8g",
    "Plank": "https://www.youtube.com/watch?v=Ehy8G39d_PM",
    "Biceps": "https://www.youtube.com/watch?v=p6YoI2IgmG0&pp=ygUeYmljZXBzIGV4Y2VyaXNlIHZpZGVvICB0cmFpbmln",
}


st.write(f"You chose to perform {selected_exercise}.")
transformer_class = exercise_transformers.get(selected_exercise)
webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=transformer_class,
        async_processing=True,
    )

st.write("### Next Steps:")
action_col1, action_col2,action_col3 = st.columns(3)

if action_col1.button("Watch Exercise"):
    st.write(f"You chose to watch {selected_exercise} exercise.")
    st.video(exercise_videos[selected_exercise])
