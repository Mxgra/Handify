import torch
import cv2
import mediapipe as mp
from NeuralNet import NeuralNetwork, landmark_dict
import numpy as np

from Spotify import play_playback, stop_playback

import json

with open('secrets.txt') as f:
    secrets = f.read()
secrets = json.loads(secrets)

labels = ['one', 'one_left',
              'two', 'two_left',
              'three', 'three_left',
              'four', 'four_left',
              'five', 'five_left',
              'rock', 'rock_left']

model = NeuralNetwork().to('cpu')
model.load_state_dict(torch.load('model.pt'))
model.eval()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        landmark_list = []
        for key, val in landmark_dict.items():
            # print(key, val)
            landmark_list.append([hand_landmarks.landmark[val].x,
                                  hand_landmarks.landmark[val].y,
                                  hand_landmarks.landmark[val].z])

        # Model prediction
        #print(landmark_list)
        #print(torch.Tensor(np.asarray(landmark_list)).shape)
        pred = model(torch.Tensor(np.asarray(landmark_list)))
        #print(pred)
        print(labels[torch.argmax(pred)])
        if 'rock' in labels[torch.argmax(pred)]:
            play_playback(device_id=secrets['device_id'])
        if 'two' in labels[torch.argmax(pred)]:
            stop_playback(device_id=secrets['device_id'])

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # waitkey is needed by imshow for resizing, redrawing, just general image processing things
    if cv2.waitKey(5) & 0xFF == 27:
      break

    #if results.multi_hand_landmarks is not None:
        #print(len(results.multi_hand_landmarks))
        #for hand_landmarks in results.multi_hand_landmarks:
            #print(hand_landmarks)
        #time.sleep(2)

cap.release()
#print(results)