import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf


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


def load_lstm():
    model_dir = "Models//LSTM.json"
    model_weights_dir = "Models//LSTM_Weights.hdf5"
    with open(model_dir, 'r') as json_file:
        json_saved_model = json_file.read()
    model = tf.keras.models.model_from_json(json_saved_model)
    model.load_weights(model_weights_dir)
    model.compile(loss = 'categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


def predict_lstm(model, frame):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    image, results = mediapipe_detection(frame, mp_holistic)
    # Draw landmarks
    draw_styled_landmarks(image, results)
    # Prediction logic

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-75:]

    if len(sequence) == 75:
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

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "Put your hand gestures in the rectangle", (5, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        cv2.putText(image, "Press 'q' to exit.", (5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        return sentence, image

