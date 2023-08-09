import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import av

import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for lunge


class LungeFormTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.IMPORTANT_LMS =  [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
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


        # Generate all columns of the data frame
        self.HEADERS = ["label"]  # Label column

        for lm in self.IMPORTANT_LMS:
            self.HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

        with open("./model/sklearn/lunge_input_scaler.pkl", "rb") as f:
            self.input_scaler = pickle.load(f)

        # Load model
        with open("./model/sklearn/lunge_stage_model.pkl", "rb") as f:
            self.stage_sklearn_model = pickle.load(f)

        with open("./model/sklearn/lunge_err_model.pkl", "rb") as f:
            self.err_sklearn_model = pickle.load(f)

        self.current_stage = ""
        self.counter = 0
        self.prediction_probability_threshold = 0.8
        self.ANGLE_THRESHOLDS = [60, 135]
        self.knee_over_toe = False
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


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

    def rescale_frame(self, frame, percent=50):
        '''
        Rescale a frame to a certain percentage compared to its original frame
        '''
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def calculate_angle(self, point1: list, point2: list, point3: list) -> float:
        '''
        Calculate the angle between 3 points
        Unit of the angle will be in Degree
        '''
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)

        # Calculate algo
        angleInRad = np.arctan2(
            point3[1] - point2[1], point3[0] - point2[0]
        ) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
        angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

        angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
        return angleInDeg

    def analyze_knee_angle(
        self, mp_results, stage: str, angle_thresholds: list, draw_to_image: tuple = None
    ):
        """
        Calculate angle of each knee while performer at the DOWN position

        Return result explanation:
            error: True if at least 1 error
            right
                error: True if an error is on the right knee
                angle: Right knee angle
            left
                error: True if an error is on the left knee
                angle: Left knee angle
        """
        results = {
            "error": None,
            "right": {"error": None, "angle": None},
            "left": {"error": None, "angle": None},
        }

        landmarks = mp_results.pose_landmarks.landmark

        # Calculate right knee angle
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        results["right"]["angle"] = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate left knee angle
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        results["left"]["angle"] = self.calculate_angle(left_hip, left_knee, left_ankle)

        # Draw to image
        if draw_to_image is not None and stage != "down":
            (image, video_dimensions) = draw_to_image

            # Visualize angles
            cv2.putText(
                image,
                str(int(results["right"]["angle"])),
                tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(int(results["left"]["angle"])),
                tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if stage != "down":
            return results

        # Evaluation
        results["error"] = False

        if angle_thresholds[0] <= results["right"]["angle"] <= angle_thresholds[1]:
            results["right"]["error"] = False
        else:
            results["right"]["error"] = True
            results["error"] = True

        if angle_thresholds[0] <= results["left"]["angle"] <= angle_thresholds[1]:
            results["left"]["error"] = False
        else:
            results["left"]["error"] = True
            results["error"] = True

        # Draw to image
        if draw_to_image is not None:
            (image, video_dimensions) = draw_to_image

            right_color = (255, 255, 255) if not results["right"]["error"] else (0, 0, 255)
            left_color = (255, 255, 255) if not results["left"]["error"] else (0, 0, 255)

            right_font_scale = 0.5 if not results["right"]["error"] else 1
            left_font_scale = 0.5 if not results["left"]["error"] else 1

            right_thickness = 1 if not results["right"]["error"] else 2
            left_thickness = 1 if not results["left"]["error"] else 2

            # Visualize angles
            cv2.putText(
                image,
                str(int(results["right"]["angle"])),
                tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                right_font_scale,
                right_color,
                right_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(int(results["left"]["angle"])),
                tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                left_font_scale,
                left_color,
                left_thickness,
                cv2.LINE_AA,
            )

        return results

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Reduce size of a frame
        image = self.rescale_frame(image, 100)
        # image = cv2.flip(image, 1)

        video_dimensions = [image.shape[1], image.shape[0]]

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.pose.process(image)

        if not results.pose_landmarks:
            print("No human found")

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
            # Extract keypoints from frame for the input
            row = self.extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=self.HEADERS[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            # Make prediction and its probability
            stage_predicted_class = self.stage_sklearn_model.predict(X)[0]
            stage_prediction_probabilities = self.stage_sklearn_model.predict_proba(X)[0]
            stage_prediction_probability = round(
                stage_prediction_probabilities[stage_prediction_probabilities.argmax()], 2
            )

            # Evaluate model prediction
            if (
                stage_predicted_class == "I"
                and stage_prediction_probability >= self.prediction_probability_threshold
            ):
                self.current_stage = "init"
            elif (
                stage_predicted_class == "M"
                and stage_prediction_probability >= self.prediction_probability_threshold
            ):
                self.current_stage = "mid"
            elif (
                stage_predicted_class == "D"
                and stage_prediction_probability >= self.prediction_probability_threshold
            ):
                if self.current_stage in ["mid", "init"]:
                    self.counter += 1

                self.current_stage = "down"

            # Error detection
            # Knee angle
            self.analyze_knee_angle(
                mp_results=results,
                stage=self.current_stage,
                angle_thresholds=self.ANGLE_THRESHOLDS,
                draw_to_image=(image, video_dimensions),
            )

            # Knee over toe
            err_predicted_class = err_prediction_probabilities = err_prediction_probability = None
            if self.current_stage == "down":
                err_predicted_class = self.err_sklearn_model.predict(X)[0]
                err_prediction_probabilities = self.err_sklearn_model.predict_proba(X)[0]
                err_prediction_probability = round(
                    err_prediction_probabilities[err_prediction_probabilities.argmax()], 2
                )

            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (800, 45), (245, 117, 16), -1)

            # Display stage prediction
            cv2.putText(
                image,
                "STAGE",
                (15, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(stage_prediction_probability),
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                self.current_stage,
                (50, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Display error prediction
            cv2.putText(
                image,
                "K_O_T",
                (200, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(err_prediction_probability),
                (195, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(err_predicted_class),
                (245, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Display Counter
            cv2.putText(
                image,
                "COUNTER",
                (110, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(self.counter),
                (110, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

        return image


def main():
    st.title("Lunge Form Correction")
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=LungeFormTransformer,
        async_processing=True,
    )


if __name__ == "__main__":
    main()
