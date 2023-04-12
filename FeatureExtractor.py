import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

landmarks_matrix = []
#print(landmarks_matrix.shape)

landmark_dict = {"WRIST": 0,
                         "THUMB_CMC": 1,
                         "THUMB_MCP": 2,
                         "THUMB_IP": 3,
                         "THUMB_TIP": 4,
                         "INDEX_FINGER_MCP": 5,
                         "INDEX_FINGER_PIP": 6,
                         "INDEX_FINGER_DIP": 7,
                         "INDEX_FINGER_TIP": 8,
                         "MIDDLE_FINGER_MCP": 9,
                         "MIDDLE_FINGER_PIP": 10,
                         "MIDDLE_FINGER_DIP": 11,
                         "MIDDLE_FINGER_TIP": 12,
                         "RING_FINGER_MCP": 13,
                         "RING_FINGER_PIP": 14,
                         "RING_FINGER_DIP": 15,
                         "RING_FINGER_TIP": 16,
                         "PINKY_MCP": 17,
                         "PINKY_PIP": 18,
                         "PINKY_DIP": 19,
                         "PINKY_TIP": 20}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = os.listdir("frames")
#IMAGE_FILES = ["frames/one_2.jpg", "frames/one_3.jpg"]
labels = []

no_results_count = 0
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread("frames/"+file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        #print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            no_results_count += 1
            continue

        #image_height, image_width, _ = image.shape
        #annotated_image = image.copy()

        #print("For loop:")

        # idx is probably for multiple recognized hands, eg. in "love" sign, we skipt that for now (see above)
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = []

            for key, val in landmark_dict.items():
                #print(key, val)
                landmark_list.append([hand_landmarks.landmark[val].x,
                                      hand_landmarks.landmark[val].y,
                                      hand_landmarks.landmark[val].z])

            landmarks_matrix.append(landmark_list)
            # add labels:
            labels.append([idx, file.split('-')[0]])
        #print(landmarks_matrix)


landmarks_matrix = np.asarray(landmarks_matrix)
print(landmarks_matrix.shape)
print(len(labels))
np.save("features", landmarks_matrix)
df = pd.DataFrame(labels, columns=["idx", "label"])
df.to_csv('annotations_file.csv', index=False)

print("No Results for " + no_results_count + " frames")
