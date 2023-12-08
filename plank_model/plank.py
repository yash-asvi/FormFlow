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

selected_voice = voices[0]

engine.setProperty('voice', "en-westindies")
engine.setProperty('rate', 125)
print(selected_voice.id)

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

        self.current_stage = ""
        self.prediction_probability_threshold = 0.6
        self.stage = None
        self.flag = 0

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
        self.frame_count=0

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

    def calculate_angle(self,a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle

        return angle

    def findPose (self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #finding height, width of the image printed
                h, w, c = img.shape
                #Determining the pixels of the landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList
            # print(angle)
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
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[11].x,landmarks[11].y]
            elbow = [landmarks[13].x,landmarks[13].y]
            wrist = [landmarks[15].x,landmarks[15].y]
            hip = [landmarks[23].x,landmarks[23].y]
            knee = [landmarks[25].x,landmarks[25].y]

            # Calculate angle
            angle1 = self.calculate_angle(shoulder, elbow, wrist)
            angle2 = self.calculate_angle(shoulder, hip, knee)

            # Visualize angle



            # Plank pose logic
            if angle1>70 and angle1<90 and angle2>130 and angle2<175:
                self.stage = "perfect"
            elif angle2<130:
                self.stage="Too High"
                print('Make your back straight. Bring your buttocks DOWN')

            elif angle2 > 175:
                self.stage="Too Low"
                print('Make your back straight. Bring your buttocks UP')
            else:
                self.stage="unkown"

            if self.flag == 0 and self.stage == "perfect" :
                self.flag = 1
            elif self.flag == 1 and self.stage == "wrong" :
                self.flag = 0
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            if self.stage == "Too Low":
                self.frame_count += 1
                if self.frame_count >= 10:
                    engine.say("Bring your back up")
                    engine.runAndWait()
                    self.frame_count = 0
            elif self.stage == "Too High":
                self.frame_count += 1
                if self.frame_count >= 10:
                    engine.say("Bring you back down")
                    engine.runAndWait()
                    self.frame_count = 0


            cv2.putText(image, "Stage", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, self.stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



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
