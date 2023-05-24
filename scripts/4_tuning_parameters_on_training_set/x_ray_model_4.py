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


def model_builder(hp):
    channels = 1 if IMG_COLOR == "grayscale" else 3

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=hp.Int("conv_1_filter", min_value=16, max_value=512, step=16),
                kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
                activation="relu",
                input_shape=(IMG_SIZE, IMG_SIZE, 1),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"),
            tf.keras.layers.Conv2D(
                filters=hp.Int("conv_2_filter", min_value=16, max_value=512, step=16),
                kernel_size=hp.Choice("conv_2_kernel", values=[3, 5]),
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"),
            tf.keras.layers.Conv2D(
                filters=hp.Int("conv_3_filter", min_value=16, max_value=512, step=16),
                kernel_size=hp.Choice("conv_3_kernel", values=[3, 5]),
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"),
            tf.keras.layers.Conv2D(
                filters=hp.Int("conv_4_filter", min_value=16, max_value=512, step=16),
                kernel_size=hp.Choice("conv_4_kernel", values=[3, 5]),
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"),
            tf.keras.layers.Conv2D(
                filters=hp.Int("conv_5_filter", min_value=16, max_value=512, step=16),
                kernel_size=hp.Choice("conv_5_kernel", values=[3, 5]),
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=hp.Int("dense_1_units", min_value=32, max_value=128, step=16),
                activation="relu",
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.experimental.RMSprop(
            hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, step=0.2)
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    return model


class HyperModel:
    def __init__(self, x_train, y_train, max_epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.max_epochs = max_epochs
        self.hypermodel = None

    def build(self):
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy", patience=10, restore_best_weights=True
        )

        hyper_band = kt.RandomSearch(
            model_builder,
            objective="val_binary_accuracy",
            max_trials=30,
            directory="hypertunning_logs",
            project_name="x_ray_hyperband",
        )

        hyper_band.search(
            self.x_train,
            self.y_train,
            batch_size=BATCH_SIZE,
            callbacks=[stop_early],
            epochs=self.max_epochs,
            validation_split=0.2,
        )

        print("\n")

        hyper_band.results_summary(1)

        print("\n")

        best_hyperparameters = hyper_band.get_best_hyperparameters()[0]

        hypermodel = hyper_band.hypermodel.build(best_hyperparameters)

        self.hypermodel = hypermodel

        return hypermodel


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

    def train(self, epochs, max_epochs, k=5):
        hypermodel = HyperModel(self.x_train, self.y_train, max_epochs)

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

            model = hypermodel.build()

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

            model.save(MODEL_DIR.joinpath(f"model_4_fold_{fold_number}.keras"))

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
