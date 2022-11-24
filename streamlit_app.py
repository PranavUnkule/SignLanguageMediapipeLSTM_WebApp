import streamlit as st
import cv2
import os
import numpy as np
import mediapipe as mp
import tempfile
from Mediapipe_LSTM_Classifier import load_lstm, draw_styled_landmarks, mediapipe_detection, extract_keypoints

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data//')
    # Actions that we try to detect
actions = np.array(['take a photo','temple','talk'])
    # Thirty videos worth of data
no_sequences = 30
    # Videos are going to be 30 frames in length
sequence_length = 75
    # Folder start
start_folder = 30
label_map = {label:num for num, label in enumerate(actions)}
    #Folder start
start_folder = 30
sequences, labels = [], []

def main():
    st.title('Indian Sign Language Translator using Mediapipe and LSTM Network')
    st.write("--Select Operations from the side bar")
    st.write("")

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
        "Use Pre-recorded video", "Live Detection"]
    choice = st.sidebar.selectbox("", activities)

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Signs Available for detection** ")
    st.sidebar.write("")
    st.sidebar.write("-- temple")
    st.sidebar.write("-- take a photo")
    st.sidebar.write("-- talk")

    if choice == "Use Pre-recorded video":
        model = load_lstm()
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        video_file = st.file_uploader(
            "Upload video", type=['mp4'])

        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.header("Recorded Video")
            # process = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            while st.button("Process"):
                webcam = cv2.VideoCapture(tfile.name)
                with mp_holistic.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.5) as holistic:
                    while webcam.isOpened():

                        # Read feed
                        ret, frame = webcam.read()
                        if ret is False:
                            st.stop()
                            break
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)
                        print(results)

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        # 2. Prediction logic
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        sequence = sequence[-75:]

                        if len(sequence) == 75:
                            sentence = []
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            print(actions[np.argmax(res)])
                            predictions.append(np.argmax(res))

                            # 3. Viz logic
                            if np.unique(predictions[-10:])[0] == np.argmax(res):
                                if res[np.argmax(res)] > threshold:

                                    if len(sentence) > 0:
                                        if actions[np.argmax(res)] != sentence[-1]:
                                            sentence.append(actions[np.argmax(res)])
                                    else:
                                        sentence.append(actions[np.argmax(res)])

                            if len(sentence) > 5:
                                sentence = sentence[-5:]

                            # Viz probabilities
                            # image = prob_viz(res, actions, image, colors)

                        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, ' '.join(sentence), (3, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.putText(image, "Put your hand gestures in the rectangle", (5, 60), cv2.FONT_HERSHEY_COMPLEX,
                                    0.5, (0, 0, 0))
                        cv2.putText(image, "Press 'q' to exit.", (5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                        try:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(image)
                        except:
                            FRAME_WINDOW.image()
            else:
                st.write('Stopped')


    if choice == "Live Detection":
        model = load_lstm()
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        st.header("Live Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        webcam = cv2.VideoCapture(0)

        while run:
            with mp_holistic.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.5) as holistic:
                while webcam.isOpened():

                    # Read feed
                    ret, frame = webcam.read()
                    if ret is False:
                        st.stop()
                        break
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-75:]

                    if len(sequence) == 75:
                        #sentence = []
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        predictions.append(np.argmax(res))

                        # 3. Viz logic
                        if np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:

                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        # Viz probabilities
                        # image = prob_viz(res, actions, image, colors)

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, "Put your hand gestures in the rectangle", (5, 60), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 0))
                    cv2.putText(image, "Press 'q' to exit.", (5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(image)
        else:
            st.write('Stopped')


if __name__ == "__main__":
    main()


