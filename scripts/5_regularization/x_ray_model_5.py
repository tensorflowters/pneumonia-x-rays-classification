import os
import pathlib
import sys
from datetime import datetime

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from sklearn.utils import class_weight

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

        class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(train_ds.y_dataset),
            y=(train_ds.y_dataset > 0.5).astype("int32").reshape(-1),
        )

        class_weights = dict(enumerate(class_weights))

        callback_stop_early = tf.keras.callbacks.EarlyStopping(
            min_delta=0.01,
            mode="max",
            monitor="val_binary_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        )

        callback_reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.3,
            monitor="binary_accuracy",
            patience=2,
            verbose=1,
            min_lr=0.000001
        )

        callback_model_save = tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR.joinpath(
                f"checkpoints/model_6_checkpoint_{datetime.utcnow().isoformat()}.keras"
            ),
            mode="max",
            monitor="val_binary_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

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
        self.callback_stop_early = callback_stop_early
        self.callback_reduce_learning_rate = callback_reduce_learning_rate
        self.class_names = class_names
        self.class_weights = class_weights
        self.fold_acc = []
        self.fold_loss = []
        self.image_size = img_size
        self.img_color = img_color
        self.interactive_reports = interactive_reports
        self.model = None
        self.model_save = callback_model_save
        self.scores = []
        self.test_ds = test_ds.normalized_dataset
        self.train_ds = train_ds.normalized_dataset
        self.x_test = test_ds.x_dataset
        self.x_train = train_ds.x_dataset
        self.y_test = test_ds.y_dataset
        self.y_train = train_ds.y_dataset

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
        model.add(tf.keras.layers.Dropout(0.1))
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
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        model.add(
            tf.keras.layers.Conv2D(
                256, (3, 3), strides=1, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        
        model.add(tf.keras.layers.Dropout(0.2))
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

    def train(self, epochs, k=5):
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
                callbacks=[
                    self.callback_stop_early,
                    self.callback_reduce_learning_rate,
                    self.model_save,
                ],
                class_weight=self.class_weights,
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

            model.save(MODEL_DIR.joinpath(f"model_5_fold_{fold_number}.keras"))

            print("\n\033[92mSaving done !\033[0m")

            plot_history(
                history,
                CHART_DIR.joinpath(
                    f"training_metrics/training_loss_and_accuracy_fold_{fold_number}.png"
                ),
                accuracy_metric="binary_accuracy",
                interactive=self.interactive_reports,
            )

            fold_number = fold_number + 1

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
