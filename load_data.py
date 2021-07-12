# Importing required modules.
import os
import random

import cv2
import numpy as np

# Path to required directories.
DIRECTORY = "C:\\Users\\Mohammed Zebin\\Desktop\\Python\\Github\\Face-Mask-Detection-Python"
IMAGE_DIR = os.path.join(DIRECTORY, "Images")

# The Categories of the images.
CLASS_NAMES = ["Mask", "No Mask"]

# The size of the images.
IMAGE_SIZE = 100


def get_training_data():
    # The training data.
    data = []

    # Looping through all images of each class.
    for label, class_name in enumerate(CLASS_NAMES):
        # Get the path of the class directory.
        class_dir = os.path.join(IMAGE_DIR, class_name)
        for img in os.listdir(class_dir):
            # Get the path of the image.
            img_path = os.path.join(class_dir, img)

            # Reading the images.
            img_array = cv2.imread(img_path)

            # Resizing the images.
            img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))

            # Appending the image and label to data.
            data.append([img_array, label])

    # Shuffle the data and return it.
    random.shuffle(data)
    return data


def split_data(data):
    images = []
    labels = []

    # Loop through the data.
    for image, label in data:
        # Append image and label to images and labels.
        images.append(image)
        labels.append(label)

    # return images and labels converted to a numpy array.
    return np.array(images) / 255, np.array(labels)


def save_data(training_images, training_labels):
    # Getting the path to files.
    training_images_file = os.path.join(DIRECTORY, "images.npy")
    training_labels_file = os.path.join(DIRECTORY, "labels.npy")

    # Writing to the files.
    np.save(training_images_file, training_images)
    np.save(training_labels_file, training_labels)


def main():
    # Getting the training data.
    training_data = get_training_data()

    # Splitting the training data.
    training_images, training_labels = split_data(training_data)

    # Saving the training data.
    save_data(training_images, training_labels)


if __name__ == "__main__":
    main()
