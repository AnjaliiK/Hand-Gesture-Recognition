#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import speech_recognition as sr

# In[2]:


import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import PIL
import numpy as np
import cv2
from PIL import ImageTk
import PIL.Image

import mediapipe as mp
from playsound import playsound
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
from gtts import gTTS
import ffmpeg
from pathlib import Path

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# initializing the mediapipe API's for drawing landmarks and for preparing the hand landmark info
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
mpHands = mp.solutions.hands

handsLandmarker = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3,
                                max_num_hands=2)  # we initialized the hand landmark detecter to detect max of 2 hands

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

offset = 20
imgSize = 300

labels = ["Dislike", "Goodjob", "Hello", "Iloveyou", "OK", "Peace"]

last_prediction = None
play = False


def audio_frame_callback(frame: av.AudioFrame):
    global play

    # if play:
    #     play = False
    #     try:
    #         # Open the MP3 file using the appropriate demuxer
    #         container = av.open("pred.mp3", mode="r")
    #
    #         # Extract the audio stream
    #         audio_stream = next(stream for stream in container.streams if stream.type == "audio")
    #
    #         # Read frames efficiently using a generator
    #         for packet in container.demux(audio_stream):
    #             for frame in packet.decode():
    #                 yield frame
    #
    #     except FileNotFoundError:
    #         print("Error: MP3 file not found.")
    #         yield None
    #     except av.AVError as err:
    #         print(f"Error decoding MP3 file: {err}")
    #         yield None
    #     finally:
    #         # Ensure container is closed even if exceptions occur
    #         if container:
    #             container.close()
    #
    # else:
    silence_frame = av.AudioFrame.empty_like(frame)
    return silence_frame


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global last_prediction, play

    image = frame.to_ndarray(format="bgr24")

    # Run inference
    dataToBePredicted = []  # this list will store 2 lists for 2 hand landmarks  [[x1, y1, x2, y2, ....x21, y21], [x1, y1, x2, y2, ....x21, y21]]. If 1 hand is detected then 1 list is stored
    bboxData = []  # this list will contain [[min(x1 coords), min(y1 coords), max(x1 coords), max(y1 coords)], [min(x1 coords), min(y1 coords), max(x1 coords), max(y1 coords)]]
    H, W, _ = image.shape
    frameOut = image.copy()

    handsLandmarkerResults = handsLandmarker.process(frameOut)

    if handsLandmarkerResults.multi_hand_landmarks:  # here we check if any hand is detected. if detected then we go inside if statement
        for handLandmarkerResult in handsLandmarkerResults.multi_hand_landmarks:  # this loop is to draw the visualization of all landmarks detected
            mpDrawing.draw_landmarks(
                frameOut,  # image to draw
                handLandmarkerResult,  # model output
                mpHands.HAND_CONNECTIONS,  # hand connections
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style())

        for handLandmarkerResult in handsLandmarkerResults.multi_hand_landmarks:  # if only 1 hand is detected then this loop execute for 1 time and if 2 detected then 2 times
            imgLandmarkData = []  # we initialized this to store x, y coords of each landmark detected for single hand.
            x_ = []  # we initiated this list to store all x coords of each landmark of single hand. so that we can get min, max values and append them to bboxData
            y_ = []  # we initiated this list to store all y coords of each landmark of single hand. so that we can get min, max values and append them to bboxData
            for landmarkIndex in range(
                    len(handLandmarkerResult.landmark)):  # for 1st hand detected, we take all x, y coords of all 21 landmarks
                x = handLandmarkerResult.landmark[landmarkIndex].x
                y = handLandmarkerResult.landmark[landmarkIndex].y
                imgLandmarkData.append(
                    x)  # we store x, y coords of all 21 landmark points to a list. this list is used for prediction using trained model
                imgLandmarkData.append(y)
                x_.append(x)
                y_.append(y)
            dataToBePredicted.append(imgLandmarkData)
            bboxData.append([min(x_), min(y_), max(x_), max(y_)])

        if len(dataToBePredicted) > 1:  # here if dataToBePredicted list length == 2 then we go in and draw 2 bounding boxes, predicted classes for 2 hands that are detected

            x1A = int(bboxData[0][0] * W) - 15
            y1A = int(bboxData[0][1] * H) - 15

            x2A = int(bboxData[0][2] * W) - 15
            y2A = int(bboxData[0][3] * H) - 15

            x1B = int(bboxData[1][0] * W) - 15
            y1B = int(bboxData[1][1] * H) - 15

            x2B = int(bboxData[1][2] * W) - 15
            y2B = int(bboxData[1][3] * H) - 15

            predictionA = model.predict([np.array(dataToBePredicted[0])])
            predictionAClass = predictionA[0]

            predictionB = model.predict([np.array(dataToBePredicted[1])])
            predictionBClass = predictionB[0]

            cv2.rectangle(frameOut, (x1A, y1A), (x2A, y2A), (0, 0, 0), 1)
            cv2.putText(frameOut, predictionAClass, (x1A, y1A - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)

            cv2.rectangle(frameOut, (x1B, y1B), (x2B, y2B), (0, 0, 0), 1)
            cv2.putText(frameOut, predictionBClass, (x1B, y1B - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)

        if len(dataToBePredicted) == 1:  # here if dataToBePredicted list length == 1 then we go in and draw 1 bounding boxes, predicted class for 1 hand1 that is detected
            x1A = int(bboxData[0][0] * W) - 8
            y1A = int(bboxData[0][1] * H) - 8

            x2A = int(bboxData[0][2] * W) + 8
            y2A = int(bboxData[0][3] * H) + 20

            predictionA = model.predict([np.array(dataToBePredicted[0])])
            predictionAClass = predictionA[0]

            cv2.rectangle(frameOut, (x1A, y1A), (x2A, y2A), (0, 0, 0), 1)
            cv2.putText(frameOut, predictionAClass, (x1A, y1A - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)

            if last_prediction != predictionAClass:
                last_prediction = predictionAClass
                # tts = gTTS(text=predictionAClass, lang='en')
                # tts.save("pred.mp3")
                try:
                    playsound(str(Path(__file__).parent) + "\\" + (predictionAClass+".mp3"),block=False)
                except Exception as e:
                    print(e)



    return av.VideoFrame.from_ndarray(frameOut, format="bgr24")


class HandGestureRecognition(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        return imgOutput

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        st.write("Recognizing...")
        text = recognizer.recognize_google(audio)
        st.write("You said:", text)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, could not understand audio.")
        return ""
    except sr.RequestError as e:
        st.write("Could not request results; {0}".format(e))
        return ""

def check_sim(word, file_map):
    for item in file_map:
        for item_word in file_map[item]:
            if word == item_word:
                return True, item
    return False, ""

def func(a, file_map, alpha_dest, op_dest):
    all_frames = []  # List to store all frames
    final = PIL.Image.new('RGB', (380, 260))  # Final image to be saved as GIF
    words = a.split()  # Split input text into words

    # Iterate through each word in the input text
    for i in words:
        flag, sim = check_sim(i, file_map)  # Check if word exists in file_map
        print(flag)
        print("flag")
        if flag == False:  # Word not found in file_map (needs to be spelled out)
            for j in i:
                im = PIL.Image.open(alpha_dest + str(j).lower() + "_small.gif")  # Load alphabet GIF
                frameCnt = im.n_frames  # Get number of frames in GIF
                for frame_cnt in range(frameCnt):
                    im.seek(frame_cnt)  # Extract frame
                    im.save("tmp.png")  # Save frame as PNG
                    img = cv2.imread("tmp.png")  # Read saved frame
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color format
                    img = cv2.resize(img, (380, 260))  # Resize frame
                    im_arr = PIL.Image.fromarray(img)  # Convert frame to PIL Image
                    for itr in range(15):  # Repeat frame to create delay
                        all_frames.append(im_arr)  # Append frame to all_frames list
        else:  # Word found in file_map (pre-captured sign language gesture)
            im = PIL.Image.open(op_dest + sim)  # Load sign language gesture GIF
            im.info.pop('background', None)  # Remove background info
            im.save('tmp.gif', 'gif', save_all=True)  # Save GIF with all frames
            im = PIL.Image.open("tmp.gif")  # Re-open saved GIF
            frameCnt = im.n_frames  # Get number of frames in GIF
            for frame_cnt in range(frameCnt):
                im.seek(frame_cnt)  # Extract frame
                im.save("tmp.png")  # Save frame as PNG
                img = cv2.imread("tmp.png")  # Read saved frame
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color format
                img = cv2.resize(img, (380, 260))  # Resize frame
                im_arr = PIL.Image.fromarray(img)  # Convert frame to PIL Image
                all_frames.append(im_arr)  # Append frame to all_frames list

    final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)  # Save frames as GIF
    return all_frames  # Return list of all frames


import streamlit as st

# Define func(a) function here...

import streamlit as st
input_text = ""

def main():
    global input_text
    st.set_page_config(
        page_title="Hand Gesture Recognition Application",
        page_icon=":wave:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "_initialized" not in st.session_state:
        st.session_state["_initialized"] = True
        st.session_state["voice_input"] = ""
    # st.session_state["_initialized"] = True
    # Hand Gesture Recognition Application
    st.title("Real Time Hand Gesture Recognition Application")
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your hand gestures")

    # Apply bright theme with white background
    st.markdown(
        """
        <style>
        body {
            color: #333333; /* Dark gray text color */
            background-color: #ffffff; /* White background */
        }
        .sidebar .sidebar-content {
            background-color: #ffffff; /* White sidebar background */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0099cc; /* Bright blue header color */
        }
        .stButton>button {
            background-color: #00cc99; /* Bright green accent color for buttons */
        }
        .stButton>button:hover {
            background-color: #00b386; /* Darker green accent color on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    webrtc_streamer(key="hand_gesture_recognition", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_frame_callback=video_frame_callback,
                    async_processing=True,
                    audio_frame_callback=audio_frame_callback)

    st.sidebar.header("About")
    st.sidebar.markdown(
        "Developed by Abhinav C, Anjali K, Ardhra Mariya, Ayaan Sameer, M V Adithya Krishnan"
    )
    
    st.sidebar.header("Model Information")
    st.sidebar.text("RandomForestClassifier Model")
    st.sidebar.text("Labels: " + ", ".join(labels))

    # Load file_map and other required variables
    op_dest = 'C:\\Users\\kansu\\OneDrive\\Desktop\\Mega-Project\\two-way-sign-language-translator\\filtered_data\\'
    alpha_dest = 'C:\\Users\\kansu\\OneDrive\\Desktop\\Mega-Project\\two-way-sign-language-translator\\alphabet\\'
    dirListing = os.listdir(op_dest)
    editFiles = []
    for item in dirListing:
        if ".webp" in item:
            editFiles.append(item)

    file_map={}
    for i in editFiles:
        tmp=i.replace(".webp","")
        tmp=tmp.split()
        file_map[i]=tmp
    
    st.header("Word to Sign Conversion")
    st.write("Enter the text below")
    # Input field for entering text

    # Create a text input box for manual input
    text_input = st.text_input("Enter text for conversion to GIF")

    voice_input = ""
    # Add a button to trigger voice input only if text input is empty
    if st.button("Click to Speak"):
        voice_input = recognize_speech() 
         # Recognize speech if button clicked
        st.session_state["voice_input"] = voice_input

        st.text_input("Voice Input:", value=voice_input)

    # Use either text input or voice input
    input_text = text_input if st.session_state["voice_input"] == "" else st.session_state["voice_input"]
    print(input_text)


    # Button to trigger conversion
    if st.button("Convert"):
        print(input_text)
        if input_text:
            # Call the func(a) function to convert text to sign language GIFs
            gif_frames = func(input_text, file_map, alpha_dest, op_dest)
            print("Exiteddddddddddddddddddddddddddddddddddd")
            # Display the resulting GIF
            st.image("out.gif", caption="Sign Language GIF", use_column_width=True)

if __name__ == "__main__":
    main()