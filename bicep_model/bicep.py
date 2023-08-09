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


# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with open("./model/bicep_curl_input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)

with open("./model/bicep_curl_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)




class BicepVideoTransformer(VideoTransformerBase):

    class BicepPoseAnalysis:
        def __init__(
            self,
            side: str,
            stage_down_threshold: float,
            stage_up_threshold: float,
            peak_contraction_threshold: float,
            loose_upper_arm_angle_threshold: float,
            visibility_threshold: float,
        ):
            # Initialize thresholds
            self.stage_down_threshold = stage_down_threshold
            self.stage_up_threshold = stage_up_threshold
            self.peak_contraction_threshold = peak_contraction_threshold
            self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
            self.visibility_threshold = visibility_threshold

            self.side = side
            self.counter = 0
            self.stage = "down"
            self.is_visible = True
            self.detected_errors = {
                "LOOSE_UPPER_ARM": 0,
                "PEAK_CONTRACTION": 0,
            }

            # Params for loose upper arm error detection
            self.loose_upper_arm = False

            # Params for peak contraction error detection
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None
        def calculate_angle(self, point1: list, point2: list, point3: list) -> float:
            '''
            Calculate the angle between 3 points
            Unit of the angle will be in Degree
            '''
            point1 = np.array(point1)
            point2 = np.array(point2)
            point3 = np.array(point3)

            # Calculate algo
            angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
            angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

            angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
            return angleInDeg


        def get_joints(self, landmarks) -> bool:
            '''
            Check for joints' visibility then get joints coordinate
            '''
            side = self.side.upper()

            # Check visibility
            joints_visibility = [
                landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
                landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
                landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility,
            ]

            is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
            self.is_visible = is_visible

            if not is_visible:
                return self.is_visible

            # Get joints' coordinates
            self.shoulder = [
                landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
                landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y,
            ]
            self.elbow = [
                landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
                landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y,
            ]
            self.wrist = [
                landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
                landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y,
            ]

            return self.is_visible

        def analyze_pose(self, landmarks, frame):
            '''
            - Bicep Counter
            - Errors Detection
            '''
            self.get_joints(landmarks)

            # Cancel calculation if visibility is poor
            if not self.is_visible:
                return (None, None)

            # * Calculate curl angle for counter
            bicep_curl_angle = int(self.calculate_angle(self.shoulder, self.elbow, self.wrist))
            if bicep_curl_angle > self.stage_down_threshold:
                self.stage = "down"
            elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
            shoulder_projection = [self.shoulder[0], 1]  # Represent the projection of the shoulder to the X axis
            ground_upper_arm_angle = int(self.calculate_angle(self.elbow, self.shoulder, shoulder_projection))

            # * Evaluation for LOOSE UPPER ARM error
            if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
                # Limit the saved frame
                if not self.loose_upper_arm:
                    self.loose_upper_arm = True
                    # save_frame_as_image(frame, f"Loose upper arm: {ground_upper_arm_angle}")
                    self.detected_errors["LOOSE_UPPER_ARM"] += 1
            else:
                self.loose_upper_arm = False

            # * Evaluate PEAK CONTRACTION error
            if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
                # Save peaked contraction every rep
                self.peak_contraction_angle = bicep_curl_angle
                self.peak_contraction_frame = frame

            elif self.stage == "down":
                # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frameFca
                if (
                    self.peak_contraction_angle != 1000
                    and self.peak_contraction_angle >= self.peak_contraction_threshold
                ):
                    # save_frame_as_image(self.peak_contraction_frame, f"{self.side} - Peak Contraction: {self.peak_contraction_angle}")
                    self.detected_errors["PEAK_CONTRACTION"] += 1

                # Reset params
                self.peak_contraction_angle = 1000
                self.peak_contraction_frame = None

            return (bicep_curl_angle, ground_upper_arm_angle)

    def __init__(self):
        # Determine important landmarks for plank
        self.IMPORTANT_LMS =  [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "RIGHT_ELBOW",
            "LEFT_ELBOW",
            "RIGHT_WRIST",
            "LEFT_WRIST",
            "LEFT_HIP",
            "RIGHT_HIP",
        ]

        self.HEADERS = ["label"]  # Label column

        for lm in self.IMPORTANT_LMS:
            self.HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

        VISIBILITY_THRESHOLD = 0.65

        # Params for counter
        STAGE_UP_THRESHOLD = 90
        STAGE_DOWN_THRESHOLD = 120

        # Params to catch FULL RANGE OF MOTION error
        PEAK_CONTRACTION_THRESHOLD = 60

        # LOOSE UPPER ARM error detection
        LOOSE_UPPER_ARM = False
        LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40

        # STANDING POSTURE error detection
        POSTURE_ERROR_THRESHOLD = 0.7
        posture = "C"

        self.left_arm_analysis = self.BicepPoseAnalysis(
            side="left",
            stage_down_threshold=STAGE_DOWN_THRESHOLD,
            stage_up_threshold=STAGE_UP_THRESHOLD,
            peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=VISIBILITY_THRESHOLD,
        )

        self.right_arm_analysis = self.BicepPoseAnalysis(
            side="right",
            stage_down_threshold=STAGE_DOWN_THRESHOLD,
            stage_up_threshold=STAGE_UP_THRESHOLD,
            peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=VISIBILITY_THRESHOLD,

        )
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

       

    

    def rescale_frame(self, frame, percent):
        '''
        Rescale a frame from OpenCV to a certain percentage compare to its original frame
        '''
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)





    def extract_important_keypoints(self, results, important_landmarks: list) -> list:
        '''
        Extract important keypoints from mediapipe pose detection
        '''
        landmarks = results.pose_landmarks.landmark

        data = []
        for lm in important_landmarks:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
        
        return np.array(data).flatten().tolist()


    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        POSTURE_ERROR_THRESHOLD = 0.7
        posture = "C"



        # Reduce size of a frame
        image = self.rescale_frame(image, 100)
        # image = cv2.flip(image, 1)

        video_dimensions = [image.shape[1], image.shape[0]]
        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.pose.process(image)

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )

        # Make detection
        try:
            landmarks = results.pose_landmarks.landmark

            (left_bicep_curl_angle, left_ground_upper_arm_angle) = self.left_arm_analysis.analyze_pose(
                landmarks=landmarks, frame=image
            )
            (right_bicep_curl_angle, right_ground_upper_arm_angle) = self.right_arm_analysis.analyze_pose(
                landmarks=landmarks, frame=image
            )

            # Extract keypoints from frame for the input
            row = self.extract_important_keypoints(results, self.IMPORTANT_LMS)
            X = pd.DataFrame([row], columns=self.HEADERS[1:])
            X = pd.DataFrame(input_scaler.transform(X))

            # Make prediction and its probability
            predicted_class = sklearn_model.predict(X)[0]
            prediction_probabilities = sklearn_model.predict_proba(X)[0]
            class_prediction_probability = round(prediction_probabilities[np.argmax(prediction_probabilities)], 2)

            if class_prediction_probability >= POSTURE_ERROR_THRESHOLD:
                posture = predicted_class

            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (500, 40), (245, 117, 16), -1)

            # Display probability
            cv2.putText(
                image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(self.right_arm_analysis.counter) if self.right_arm_analysis.is_visible else "UNK",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display Left Counter
            cv2.putText(
                image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(self.left_arm_analysis.counter) if self.left_arm_analysis.is_visible else "UNK",
                (100, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # * Display error
            # Right arm error
            cv2.putText(
                image, "R_PC", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(self.right_arm_analysis.detected_errors["PEAK_CONTRACTION"]),
                (160, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image, "R_LUA", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(self.right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]),
                (220, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Left arm error
            cv2.putText(
                image, "L_PC", (300, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(self.left_arm_analysis.detected_errors["PEAK_CONTRACTION"]),
                (295, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image, "L_LUA", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(self.left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]),
                (375, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Lean back error
            cv2.putText(
                image, "LB", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                str(f"{posture}, {predicted_class}, {class_prediction_probability}"),
                (440, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # * Visualize angles
            # Visualize LEFT arm calculated angles
            if self.left_arm_analysis.is_visible:
                cv2.putText(
                    image,
                    str(left_bicep_curl_angle),
                    tuple(np.multiply(self.left_arm_analysis.elbow, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(left_ground_upper_arm_angle),
                    tuple(np.multiply(self.left_arm_analysis.shoulder, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Visualize RIGHT arm calculated angles
            if self.right_arm_analysis.is_visible:
                cv2.putText(
                    image,
                    str(right_bicep_curl_angle),
                    tuple(np.multiply(self.right_arm_analysis.elbow, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(right_ground_upper_arm_angle),
                    tuple(np.multiply(self.right_arm_analysis.shoulder, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        except Exception as e:
            print(f"Error: {e}")

        return image


def main():
    st.title("Bicep Form Correction")
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=BicepVideoTransformer,
        async_processing=True,
    )


if __name__ == "__main__":
    main()
