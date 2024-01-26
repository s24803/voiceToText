from transformers import pipeline
import streamlit as st
import time
from pydub import AudioSegment

st.set_page_config(
    page_title="Voice to text",
    layout="wide"
)

st.title("Generate text for your video 🤩")


@st.cache_resource
def load_model(model_name):
    return pipeline("automatic-speech-recognition", model=model_name)


def video_to_audio(file):
    video = AudioSegment.from_file(file=file)
    return video.export(format="wav").read()


whisper = load_model("openai/whisper-base.en")

uploaded_file = st.file_uploader("Choose a file", type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'])

result = None

if st.button("Generate text") and uploaded_file is not None:
    audio = video_to_audio(uploaded_file)
    result = whisper(audio)
    st.write(result["text"])

if result is not None:
    st.download_button("Download text", result["text"])

