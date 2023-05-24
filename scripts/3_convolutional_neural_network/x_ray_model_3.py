import os
import pathlib
import sys

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.x_ray_data_viz import plot_history
from utils.x_ray_dataset_builder import Dataset

MODEL_ID = os.getenv("MODEL_ID")
BATCH_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_BATCH_SIZE"))
CHART_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_CHART_DIR")).absolute()
MODEL_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_MODEL_DIR")).absolute()
IMG_COLOR = os.getenv(f"MODEL_{MODEL_ID}_IMG_COLOR")
IMG_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_IMG_SIZE"))


class Model:
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        img_color: str,
        label_mode: str,
        interactive_reports: bool = True,
    ):
        train_dir = pathlib.Path("data/train")

        test_dir = pathlib.Path("data/test")

        train_ds = Dataset(
            train_dir,
            batch_size=batch_size,
            color_mode=img_color,
            image_size=img_size,
            label_mode=label_mode,
        )

        test_ds = Dataset(
            test_dir,
            batch_size=batch_size,
            color_mode=img_color,
            image_size=img_size,
            label_mode=label_mode,
        )

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds.build(AUTOTUNE, True)

        test_ds.build(AUTOTUNE)

        class_names = train_ds.get_class_names()
        print("\nClass names:")
        print(class_names)

        train_x_batch_shape = train_ds.get_x_batch_shape()
        print("\nTraining dataset's images batch shape is:")
        print(train_x_batch_shape)

        train_y_batch_shape = train_ds.get_y_batch_shape()
        print("\nTraining dataset's labels batch shape is:")
        print(train_y_batch_shape)

        train_ds.display_images_in_batch(
            1,
            "Training dataset",
            CHART_DIR.joinpath("dataset/train_dataset_image_sample.png"),
            interactive=interactive_reports,
        )
        train_ds.display_batch_number(
            "Training dataset",
            CHART_DIR.joinpath("dataset/train_dataset_batch_number.png"),
            interactive=interactive_reports,
        )

        test_x_batch_shape = train_ds.get_x_batch_shape()
        print("\nTesting dataset's images batch shape is:")
        print(test_x_batch_shape)

        test_y_batch_shape = train_ds.get_y_batch_shape()
        print("\nTesting dataset's labels batch shape is:")
        print(test_y_batch_shape)

        test_ds.display_images_in_batch(
            1,
            "Testing dataset",
            CHART_DIR.joinpath("dataset/test_dataset_image_sample.png"),
            interactive=interactive_reports,
        )
        test_ds.display_batch_number(
            "Testing dataset",
            CHART_DIR.joinpath("dataset/test_dataset_batch_number.png"),
            interactive=interactive_reports,
        )

        self.batch_size = batch_size
        self.class_names = class_names
        self.fold_acc = []
        self.fold_loss = []
        self.image_size = img_size
        self.img_color = img_color
        self.interactive_reports = interactive_reports
        self.scores = []
        self.train_ds = train_ds.normalized_dataset
        self.test_ds = test_ds.normalized_dataset
        self.x_train = train_ds.x_dataset
        self.x_test = test_ds.x_dataset
        self.y_train = train_ds.y_dataset
        self.y_test = test_ds.y_dataset
        
    def build(self):
        channels = 1 if IMG_COLOR == "grayscale" else 3

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=1,
                padding="same",
                activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, channels)
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        model.add(
            tf.keras.layers.Conv2D(
                64, (3, 3), strides=1, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        model.add(
            tf.keras.layers.Conv2D(
                64, (3, 3), strides=1, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        model.add(
            tf.keras.layers.Conv2D(
                128, (3, 3), strides=1, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        model.add(
            tf.keras.layers.Conv2D(
                256, (3, 3), strides=1, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        
        model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

        model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        
        return model


    def train(self, epochs, max_epochs, k=5):
        kfold = KFold(n_splits=k, shuffle=True)

        fold_number = 1

        for train_index, val_index in kfold.split(self.x_train, self.y_train):
            print(
                "\033[91m"
                "=================================================================\n"
                f"**************STARTING TRAINING K-FOLD N°{fold_number}***********\n"
                "================================================================="
                "\033[0m"
            )

            train_images, val_images = (
                self.x_train[train_index],
                self.x_train[val_index],
            )

            train_labels, val_labels = (
                self.y_train[train_index],
                self.y_train[val_index],
            )

            print("\033[96mBuilding model...\n")

            model = self.build()

            history = model.fit(
                train_images,
                train_labels,
                batch_size=self.batch_size,
                epochs=epochs,
                validation_data=(val_images, val_labels),
            )

            self.scores = model.evaluate(
                self.x_test, self.y_test, batch_size=self.batch_size, verbose=1
            )

            self.fold_acc.append(self.scores[1] * 100)

            self.fold_loss.append(self.scores[0])

            print("\033[0m")

            print(
                "\n\033[91m"
                "=================================================================\n"
                f"**************TRAINING FOR K-FOLD N°{fold_number} DONE!**********\n"
                "================================================================="
                "\033[0m"
            )

            print(f"\n\033[91mSaving model n°{fold_number}...\033[0m")

            model.save(MODEL_DIR.joinpath(f"model_3_fold_{fold_number}.keras"))

            print("\n\033[92mSaving done !\033[0m")

            plot_history(
                history,
                CHART_DIR.joinpath(
                    f"training_metrics/training_loss_and_accuracy_fold_{fold_number}.png"
                ),
                accuracy_metric="binary_accuracy",
                interactive=self.interactive_reports,
            )

            fold_number += 1

        print("\nScore per fold")

        for i in range(0, len(self.fold_acc)):
            print(
                "\n------------------------------------------------------------------------"
            )
            print(
                f"> Fold {i+1} - Loss: {self.fold_loss[i]} - Accuracy: {self.fold_acc[i]}%"
            )

        print(
            "\n\033[92m"
            "=================================================================\n"
            "********************AVERAGE SCORES FOR ALL FOLDS*****************\n"
            "================================================================="
            "\033[0m\n"
        )

        print(
            f"> Average accuracy: {np.mean(self.fold_acc)} (+- {np.std(self.fold_acc)})"
        )

        print(f"> Average loss: {np.mean(self.fold_loss)}")

        print(
            "\n\033[91m"
            "=================================================================\n"
            "****************************TRAINING DONE************************\n"
            "================================================================="
            "\033[0m\n"
        )
