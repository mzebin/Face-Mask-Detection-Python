# Importing required modules.
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Path to required directories.
DIRECTORY = "C:\\Users\\Mohammed Zebin\\Desktop\\Python\\Github\\Face-Mask-Detection-Python"


def load_data():
    # Load the images and labels file.
    images_file = os.path.join(DIRECTORY, "images.npy")
    labels_file = os.path.join(DIRECTORY, "labels.npy")

    images = np.load(images_file)
    labels = np.load(labels_file)

    return images, labels


def create_model(training_data_shape):
    # Creating model.
    model = models.Sequential()

    # Adding Layers.
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, input_shape=training_data_shape, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))

    return model


def main():
    # Loading the training data.
    images, labels = load_data()

    training_images, testing_images, training_labels, testing_labels = train_test_split(
        images,
        labels,
        test_size=0.1,
    )

    # Creating the model.
    model = create_model(training_images.shape[1:])

    # Compiling the model.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Training the model.
    model.fit(
        training_images,
        training_labels,
        epochs=20,
        validation_data=(testing_images, testing_labels),
    )

    # Getting the results.
    loss, accuracy = model.evaluate(testing_images, testing_labels)

    # Displaying the results.
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Saving the model.
    model_file = os.path.join(DIRECTORY, "mask_detection.model")
    model.save(model_file)


if __name__ == "__main__":
    main()
