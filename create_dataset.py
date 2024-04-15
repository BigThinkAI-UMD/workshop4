# Import the os module for operating system functionality
import os
# Import pickle module for serializing data
import pickle

# Import the mediapipe library for AI/ML tasks
import mediapipe as mp
# Import the OpenCV library for image processing
import cv2
# Import pyplot for plotting graphs (not used in this snippet)
import matplotlib.pyplot as plt

# Shortcut to access the hand detection model
mp_hands = mp.solutions.hands
# Utility for drawing on images (not used in this snippet)
mp_drawing = mp.solutions.drawing_utils
# Styling for drawings (not used in this snippet)
mp_drawing_styles = mp.solutions.drawing_styles

# Configure hand detection model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define directory where data is stored
DATA_DIR = './data'

# List to store processed data
data = []
# List to store labels of data
labels = []
# Loop through directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    # Loop through images in each directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Temporary list to store data for one image
        data_aux = []

        # List to store x coordinates of landmarks
        x_ = []
        # List to store y coordinates of landmarks
        y_ = []

        # Read image file
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert image to RGB color space
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image using the hand detection model
        results = hands.process(img_rgb)
        # Check if any hand landmarks were detected
        if results.multi_hand_landmarks:
            # Loop through detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through individual landmarks
                for i in range(len(hand_landmarks.landmark)):
                    # Get x coordinate of landmark
                    x = hand_landmarks.landmark[i].x
                    # Get y coordinate of landmark
                    y = hand_landmarks.landmark[i].y

                    # Append x coordinate to list
                    x_.append(x)
                    # Append y coordinate to list
                    y_.append(y)

                # Loop again to calculate normalized positions
                for i in range(len(hand_landmarks.landmark)):
                    # Get x coordinate of landmark
                    x = hand_landmarks.landmark[i].x
                    # Get y coordinate of landmark
                    y = hand_landmarks.landmark[i].y
                    # Normalize x by subtracting the smallest x
                    data_aux.append(x - min(x_))
                    # Normalize y by subtracting the smallest y
                    data_aux.append(y - min(y_))

            # Append the normalized data to the main list
            data.append(data_aux)
            # Append the directory name (label) to the labels list
            labels.append(dir_)

# Print the list of labels
print(labels)

# Open a file for writing in binary mode
f = open('data.pickle', 'wb')
# Serialize the data and labels into the file
pickle.dump({'data': data, 'labels': labels}, f)
# Close the file
f.close()
