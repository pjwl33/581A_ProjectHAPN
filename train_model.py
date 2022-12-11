# Import TensorFlow and other necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import random

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from keras.losses import SparseCategoricalCrossentropy

# Import for predictions
import os
from os import listdir

# Supressing warning messages on output
import warnings
warnings.filterwarnings("ignore")

# Visualize the Training Results in Plot Data
def visualization_report(history, epochs, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val. Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Train and Val. Accuracy: {model_name}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val. Loss')
    plt.legend(loc='upper right')
    plt.title(f'Train and Val. Loss: {model_name}')
    plt.show()


def run_predictions(model, class_names, img_height, img_width, config):
    # Running Predictions on Images outside of Dataset in General
    pred_path = config["prediction_data_path"]
    plt.figure(figsize=(10, 10))
    figure_counter = 1

    for img_p in os.listdir(pred_path):
        if (img_p.endswith(".png")) or (img_p.endswith(".jpg")) or (img_p.endswith(".jpeg")):
            print(img_p)
            # Grayscale the image in the same way for prediction
            img_path = f'{pred_path}{img_p}'
            img = tf.keras.utils.load_img(
                img_path, color_mode='grayscale', target_size=(img_height, img_width), keep_aspect_ratio=True
            )

            ax = plt.subplot(3, 3, figure_counter)
            plt.imshow(img)
            plt.axis("off")

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            result_msg = f'{class_names[np.argmax(score)]}, {100 * np.max(score)}'
            plt.title(result_msg)
            print(
                "{} most likely belongs to {} with a {:.2f} percent confidence."
                .format(img_p, class_names[np.argmax(score)], 100 * np.max(score))
            )

            figure_counter += 1


def train_keras_model(config={}):

    # Requires Config
    if not bool(config):
        return None

    # Image Loader Params
    shape_size = 1
    dataset_url = config["dataset_path"]
    batch_size = config["batch_size"]
    batch_seed = config["batch_seed"]
    img_height = config["img_height"]
    img_width = config["img_width"]
    img_shape = (img_width, img_height)
    input_shape = (img_width, img_height, shape_size)

    tf_autotune = tf.data.AUTOTUNE
    data_dir = pathlib.Path(dataset_url)

    # Train Split at 70%
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        image_size=img_shape,
        seed=batch_seed,
        color_mode='grayscale',
        batch_size=batch_size)

    # Validation of the data at 70%
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        image_size=img_shape,
        seed=batch_seed,
        color_mode='grayscale',
        batch_size=batch_size)

    # Classification of names - each image type in its own dir
    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf_autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=tf_autotune)

    # Create basic Keras Model

    # The Keras Sequential model consists of three convolution blocks(tf.keras.layers.Conv2D)
    # with a max pooling layer(tf.keras.layers.MaxPooling2D) in each of them.
    #
    # There's a fully-connected layer (tf.keras.layers.Dense) with 128 units on top of it that
    # is activated by a ReLU activation function ('relu').
    #
    # This model has not been tuned for high accuracy; the goal of this tutorial is to show a standard approach.

    num_classes = len(class_names)
    model_save_path = config["model_save_path"]
    model_save_name = config["model_save_name"]
    epochs = config["epochs"]

    # try:
    # model = keras.models.load_model(model_save_path)
    # history = model.history
    # except OSError:
    model = Sequential([
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(16, shape_size, activation='relu',
               padding='same', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(32, shape_size, activation='relu',
               padding='same', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(64, shape_size, activation='relu',
               padding='same', input_shape=input_shape),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    # Compile the model

    # For this tutorial, choose the tf.keras.optimizers.Adam optimizer and
    # tf.keras.losses.SparseCategoricalCrossentropy loss function.
    #
    # To view training and validation accuracy for each training epoch, pass the metrics argument to Model.compile.
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])

    history = model.fit_generator(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save(model_save_path)
    print(f'Model Summary: {model_save_name}', model.summary())

    visualization_report(history, epochs, model_save_name)
    run_predictions(model, class_names, img_height, img_width, config)

# All different types of config to be generated


def generate_configs(model_type):
    # Generate configs by model type/name
    return {
        "dataset_path": f'data/training_{model_type}/',
        "model_save_path": f'model/trained_{model_type}',
        "model_save_name": f'trained_{model_type}',
        "prediction_data_path": f'data/predictions/{model_type}/',
        "epochs": 10,
        "batch_size": 64,
        "batch_seed": random.randint(1, 581),
        "img_height": 120,
        "img_width": 128
    }


# Human vs. Nonhuman Configurations
config_human_nonhuman = generate_configs("human_nonhuman")

# Mask vs. No Mask Configurations
config_mask_nomask = generate_configs("mask_nomask")

# Glasses vs No Glasses Configurations
config_glasses_noglasses = generate_configs("glasses_noglasses")

train_keras_model(config_human_nonhuman)
train_keras_model(config_mask_nomask)
train_keras_model(config_glasses_noglasses)
