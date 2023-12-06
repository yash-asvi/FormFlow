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

import warnings
warnings.filterwarnings('ignore')
from streamlit_lottie import st_lottie
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
def load(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

lottie=load("https://lottie.host/67658b63-5ca8-4612-b9b0-32be1deec781/FicQrmGPRj.json")

exercise_col, animation_col = st.columns([2, 1])

# Exercise selection column
with exercise_col:
    st.write("## Select an Exercise")

    # List of exercise options
    exercise_options = ["Lunge","Squat","Plank","Biceps"]
    selected_exercise = st.selectbox("Choose an exercise:", exercise_options)
    


    # Display buttons for exercise styles
    

# Animation column
# with animation_col:
#     # st.write("### Background Exercise Animation")
#     # lottie_url = "https://assets7.lottiefiles.com/packages/lf20_zyojnpzk.json"
#     st_lottie(lottie, speed=1, width=300, height=200,key="Excercise")
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


st.write(f"You chose to perform {selected_exercise} .")
transformer_class = exercise_transformers.get(selected_exercise)
webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=transformer_class,
        async_processing=True,
        rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]

    )

st.write("### Next Steps:")
action_col1, action_col2,action_col3 = st.columns(3)

if action_col1.button("Watch Exercise"):
    st.write(f"You chose to watch {selected_exercise} exercise.")
    st.video(exercise_videos[selected_exercise])
