import logging
import queue
import os
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp

# Core components
from core.capturer import SignCapturer
from core.sentence import SentenceTranslator
from core.translator import SignModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger(__name__)

# --- MEDIA PIPE UTILS ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- CONFIGURATION ---
SKELETON_PATH = "assets/10042041.parquet"
PREDICTION_WINDOW = 45 
CONFIDENCE_THRESHOLD = 0.6 

@st.cache_resource
def load_models():
    try:
        capturer = SignCapturer(skeleton_path=SKELETON_PATH)
    except FileNotFoundError:
        st.error("Please place your '10042041.parquet' file inside the 'assets' folder.")
        st.stop()
    
    sign_model = SignModel("assets/model.tflite")
    translator = SentenceTranslator()
    return capturer, sign_model, translator

capturer, sign_model, translator = load_models()
word_queue = queue.Queue() 

class SessionState:
    def __init__(self):
        self.sequence_buffer = []

state = SessionState()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="rgb24") 

    results, landmarks_df = capturer.process_frame(img, len(state.sequence_buffer))
    
    # 2. Accumulate Data
    state.sequence_buffer.append(landmarks_df)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    # Draw Hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION, # <--- Key Change: Lips only
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )

    # 4. Prediction Logic
    if len(state.sequence_buffer) >= PREDICTION_WINDOW:
        sign, conf = sign_model.predict(state.sequence_buffer)
        if conf > CONFIDENCE_THRESHOLD:
            word_queue.put((sign, conf))
        state.sequence_buffer = []

    return av.VideoFrame.from_ndarray(img, format="rgb24")

# --- UI LAYOUT ---
st.header("✨ Real-time MediaPipe Holistic Estimation")

webrtc_ctx = webrtc_streamer(
    key="asl-translator",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

sentence_placeholder = st.empty()
buffer_placeholder = st.empty()

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Full Sentence (SPACE)"):
        translator.generate_sentence()
with col2:
    if st.button("Clear Buffer"):
        translator.word_buffer = []
        translator.current_sentence = ""

if webrtc_ctx.state.playing:
    while True:
        try:
            sign, conf = word_queue.get(timeout=1.0)
            translator.add_word(sign)
            sentence_placeholder.info(f"**Final Sentence:** {translator.current_sentence}")
            buffer_placeholder.success(f"**Keyword Buffer:** {translator.get_display_string()}")
        except queue.Empty:
            sentence_placeholder.info(f"**Final Sentence:** {translator.current_sentence}")
            buffer_placeholder.success(f"**Keyword Buffer:** {translator.get_display_string()}")
            continue