import streamlit as st
import os
import imageio # type: ignore
import tensorflow as tf
import numpy as np
from utils import load_data, num_to_char
from modelutil import load_model
import subprocess
from moviepy.editor import VideoFileClip # type: ignore

#set the layout to the streamlit app as wide
st.set_page_config(layout="wide")

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip reading neural network for live transcription')
    st.info('This app is developed by Aaranay Aadi, from the paper "Lip Reading Sentences in the Wild"')

st.title("Lip reading app by RNA")
#Generating a list of options or videos
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Select a video', options)

#Generate 2 columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        input_file_path = os.path.join('..', 'data', 's1', selected_video)
        output_file_path = 'test_video.mp4'

        # Convert video using moviepy
        clip = VideoFileClip(input_file_path)
        clip.write_videofile(output_file_path, codec='libx264')

        # Rendering inside of the app
        video_file = open(output_file_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_file.close()

    with col2:
        st.info('This is what the ML model sees before making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(input_file_path))
        if video.ndim == 3:
            # Convert each grayscale frame to RGB
            video_rgb = np.stack([np.repeat(frame[:, :, np.newaxis], 3, axis=2) for frame in video], axis=0).astype(np.uint8)
        elif video.ndim == 4 and video.shape[-1] == 1:
            # Handle case where video has shape (num_frames, height, width, 1)
            video_rgb = np.repeat(video, 3, axis=-1).astype(np.uint8)
        else:
            # Assuming video is already in the correct shape and data type
            video_rgb = video.astype(np.uint8)

        # Debugging information
        print(f"Final video shape: {video_rgb.shape}")
        print(f"Data type: {video_rgb.dtype}")

        # Save the video as a GIF, ensuring the file path is correct and accessible
        try:
            imageio.mimsave('animation.gif', video_rgb, fps=10)
            print("GIF saved successfully.")
        except Exception as e:
            print(f"Error saving GIF: {e}")

        st.image('animation.gif', use_column_width=True)

        st.info('This is the output of the model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0]
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)