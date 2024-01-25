from transformers import pipeline
import streamlit as st


whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

result = whisper("nice.mp3")

print(result)
# st.set_page_config(
#     page_title="Text-It",
#     layout="wide"
# )
#
# st.title("Generate text for your video 🤩")
#
# uploaded_file = st.file_uploader("Choose a file")
#
# if uploaded_file is not None:
#     print("hello")
