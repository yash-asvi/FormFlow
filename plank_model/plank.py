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

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for plank



class PlankPoseEstimator(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        # Load model
        with open("./model/plank_model.pkl", "rb") as f:
            self.sklearn_model = pickle.load(f)

        # Dump input scaler
        with open("./model/plank_input_scaler.pkl", "rb") as f2:
            self.input_scaler = pickle.load(f2)

        self.current_stage = ""
        self.prediction_probability_threshold = 0.6

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.IMPORTANT_LMS = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
            "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX",
        ]

        self.HEADERS = ["label"]  # Label column

        for lm in self.IMPORTANT_LMS:
            self.HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    def extract_important_keypoints(self, results) -> list:
        '''
        Extract important keypoints from mediapipe pose detection
        '''
        landmarks = results.pose_landmarks.landmark

        data = []
        for lm in self.IMPORTANT_LMS:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

        return np.array(data).flatten().tolist()


    def rescale_frame(self, frame, percent=100):
        '''
        Rescale a frame to a certain percentage compared to its original frame
        '''
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)



    def transform(self, frame):

        image = frame.to_ndarray(format="bgr24")
        # Reduce size of a frame
        image = self.rescale_frame(image, 100)
        # image = cv2.flip(image, 1)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            return image

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

        # Make detection
        try:
            # Extract keypoints from frame for the input
            row = self.extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=self.HEADERS[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            # Make prediction and its probability
            predicted_class = self.sklearn_model.predict(X)[0]
            prediction_probability = self.sklearn_model.predict_proba(X)[0]
            print(predicted_class, prediction_probability)

            # Evaluate model prediction
            if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= self.prediction_probability_threshold:
                self.current_stage = "Correct"
            elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= self.prediction_probability_threshold:
                self.current_stage = "Low back"
            elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= self.prediction_probability_threshold:
                self.current_stage = "High back"
            else:
                self.current_stage = "unk"

            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display class
            cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, self.current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display probability
            cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability[np.argmax(prediction_probability)], 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        return image


def main():
    st.title("Plank Form Correction")
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=PlankPoseEstimator,  # Use the custom transformer
        async_processing=True,
    )


if __name__ == "__main__":
    main()
