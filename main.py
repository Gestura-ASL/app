import logging
import queue
from typing import List, NamedTuple

import av
import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc
import mediapipe as mp

logger = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic 

class LandmarkData(NamedTuple):
    area: str
    landmark_name: str
    x: float
    y: float
    visibility: float


@st.cache_resource
def get_holistic_model():
    """Initializes and returns the MediaPipe Holistic model with specified params."""
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1, # Use slider value
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

st.header("✨ Real-time MediaPipe Holistic Estimation")

holistic_detector = get_holistic_model()
min_confidence = st.slider("Min Landmark Visibility Confidence", 0.0, 1.0, 0.5, 0.05)


result_queue: "queue.Queue[List[LandmarkData]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Processes video frames using MediaPipe Holistic, draws all landmarks (Pose, Face, Hands),
    and queues the data.
    """
    image = frame.to_ndarray(format="rgb24")

    # 1. Prepare Image
    image_rgb = image
    image_rgb.flags.writeable = False

    results = holistic_detector.process(image_rgb)

    image.flags.writeable = True

    landmarks_data = []

    # --- POSE Landmarks ---
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        # Collect pose data (e.g., first 5 for the table)
        for idx, landmark in enumerate(results.pose_landmarks.landmark[:5]):
            if landmark.visibility > min_confidence:
                landmarks_data.append(
                    LandmarkData(
                        area="Pose",
                        landmark_name=mp_holistic.PoseLandmark(idx).name,
                        x=round(landmark.x, 3), y=round(landmark.y, 3),
                        visibility=round(landmark.visibility, 3)
                    )
                )

    # --- LEFT HAND Landmarks ---
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # --- RIGHT HAND Landmarks ---
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # --- FACE Landmarks ---
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
    )
    
    # Put the extracted data into the queue
    result_queue.put(landmarks_data)

    # Return the annotated frame to the client
    return av.VideoFrame.from_ndarray(image, format="rgb24")


webrtc_ctx = webrtc_streamer(
    key="mediapipe-holistic-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show a sample of detected landmarks", value=True):
    if webrtc_ctx.state.playing:
        st.subheader("Sample Detected Landmarks (Pose)")
        labels_placeholder = st.empty()
        
        while True:
            result = result_queue.get() 
            
            if result:
                data_for_table = [r._asdict() for r in result]
                labels_placeholder.dataframe(data_for_table, width="stretch")
            else:
                labels_placeholder.markdown("*(No landmarks detected or visible below threshold)*")

st.markdown("---")
st.markdown(
    f"This demo uses **MediaPipe Holistic**  for full-body tracking."
)

st.markdown(
    f"Streamlit version: {st.__version__}  \n"
    f"Streamlit-WebRTC version: {st_webrtc_version}  \n"
    f"aiortc version: {aiortc.__version__}  \n"
)