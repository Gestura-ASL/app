import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

class SignCapturer:
    def __init__(self, skeleton_path):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Using model_complexity=1 to match original accuracy (0 is faster but dumber)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        if not os.path.exists(skeleton_path):
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
            
        # Load the exact skeleton structure from the file
        xyz = pd.read_parquet(skeleton_path)
        self.xyz_skel = (
            xyz[['type', 'landmark_index']]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )
        print(f"Skeleton loaded. Rows: {len(self.xyz_skel)}")

    def process_frame(self, image, frame_number):
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

        landmarks_df = self._create_frame_landmark_df(results, frame_number)
        return results, landmarks_df

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
    def _create_frame_landmark_df(self, results, frame):
        # 1. Extraction mapping
        # Define which results map to which type
        mapping = [
            (results.face_landmarks, 'face'),
            (results.pose_landmarks, 'pose'),
            (results.left_hand_landmarks, 'left_hand'),
            (results.right_hand_landmarks, 'right_hand')
        ]
        
        all_data = []
        
        # 2. Extract raw data using a fast list comprehension
        for lms, label in mapping:
            if lms:
                for i, p in enumerate(lms.landmark):
                    all_data.append({
                        'landmark_index': i,
                        'x': p.x,
                        'y': p.y,
                        'z': p.z,
                        'type': label
                    })

        # 3. Create a single DataFrame (Avoids concat overhead and FutureWarnings)
        if all_data:
            landmarks = pd.DataFrame(all_data)
        else:
            # Fallback if nothing is detected
            landmarks = pd.DataFrame(columns=['landmark_index', 'x', 'y', 'z', 'type'])

        # 4. Merge with Skeleton
        # Aligns detected points with the full model skeleton
        landmarks = self.xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
        
        return landmarks.assign(frame=frame)