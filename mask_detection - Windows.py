# Importing required modules.
import os
import time
import threading
import winsound

import cv2
import numpy as np
from tensorflow.keras import models

# PATHS
DIRECTORY = "C:\\Users\\Mohammed Zebin\\Desktop\\Python\\Github\\Face-Mask-Detection-Python"
MODEL_PATH = os.path.join(DIRECTORY, "mask_detection.model")
CASCADE_PATH = os.path.join(DIRECTORY, "haarcascade_frontalface_default.xml")

# Creating the cascade for face recognition.
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

# Loading the model.
MODEL = models.load_model(MODEL_PATH)

# Setting the video source to the webcam.
VIDEO_CAPTURE = cv2.VideoCapture(0)

# Types of Images.
CLASS_NAMES = ["Mask", "No Mask"]
COLORS = [(0, 255, 0), (0, 0, 255)]

# Detected Colors
DETECTED_COLORS = []


def detect_faces(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return FACE_CASCADE.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )


def get_cropped_images(image, faces):
    # Looping through the faces.
    cropped_images = []
    for x, y, w, h in faces:
        # Append cropped image to cropped images.
        cropped_images.append(image[y:y + h, x:x + w])

    return cropped_images


def get_resized_images(images, size):
    return list(map(lambda img: cv2.resize(img, (size, size)), images))


def mask_detection(images):
    # Predicting the result.
    predictions = MODEL.predict(images)

    # Looping through all the predictions and
    # Appending the index to indexes.
    indexes = []
    for prediction in predictions:
        indexes.append(np.argmax(prediction))

    return indexes


def get_class_data(results):
    # Looping through all the results.
    class_names = []
    colors = []
    for result in results:
        class_names.append(CLASS_NAMES[result])
        colors.append(COLORS[result])

    return class_names, colors


def mark_faces(image, faces, colors, class_names):
    # Looping through all the faces and drawing rectangles.
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[idx], 3)
        cv2.putText(
            image,
            class_names[idx],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[idx],
            2,
        )


def warn_user():
    while True:
        # Gving a Warning to the  user
        # if red color is in detected colors.
        if COLORS[1] in DETECTED_COLORS:
            winsound.Beep(2500, 1000)

        time.sleep(1)


def main():
    global DETECTED_COLORS

    while True:
        # Capturing the frame.
        _, frame = VIDEO_CAPTURE.read()

        # Detecting faces.
        faces = detect_faces(frame)

        # If no faces are detected; continue.
        if len(faces) == 0:
            # Showing the Frame.
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            else:
                continue

        # Cropping the image.
        cropped_images = get_cropped_images(frame, faces)

        # Resizing Images.
        resized_images = get_resized_images(cropped_images, 100)

        # Detecting masks.
        results = mask_detection(np.array(resized_images) / 255)

        # Getting the colors.
        class_names, DETECTED_COLORS = get_class_data(results)

        # Marking the faces.
        mark_faces(frame, faces, DETECTED_COLORS, class_names)

        # Showing the Frame.
        cv2.imshow("Video", frame)

        # Clear the Detected Colors.
        DETECTED_COLORS.clear()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    # Creating and the Starting the threads.
    t1 = threading.Thread(target=main)
    t2 = threading.Thread(target=warn_user, daemon=True)

    t1.start()
    t2.start()
