import os
import pathlib
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.x_ray_data_viz import plot_history
from utils.x_ray_dataset_builder import Dataset

MODEL_ID = os.getenv("MODEL_ID")
CHARTS_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_CHART_DIR")).absolute()
MODELS_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_MODEL_DIR")).absolute()


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

        train_ds = Dataset(
            train_dir,
            batch_size=batch_size,
            color_mode=img_color,
            image_size=img_size,
            label_mode=label_mode,
            subset="training",
            validation_split=0.2,
        )

        val_ds = Dataset(
            train_dir,
            batch_size=batch_size,
            color_mode=img_color,
            image_size=img_size,
            label_mode=label_mode,
            subset="validation",
            validation_split=0.2,
        )

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds.build(AUTOTUNE, True)

        val_ds.build(AUTOTUNE)

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
            2,
            "Training dataset",
            CHARTS_DIR.joinpath("dataset/train_dataset_image_sample.png"),
            interactive=interactive_reports,
        )
        train_ds.display_batch_number(
            "Training dataset",
            CHARTS_DIR.joinpath("dataset/train_dataset_batch_number.png"),
            interactive=interactive_reports,
        )
        train_ds.display_distribution(
            "Training dataset",
            CHARTS_DIR.joinpath("dataset/train_dataset_image_distribution.png"),
            interactive=interactive_reports,
        )
        train_ds.display_mean(
            "Training dataset",
            CHARTS_DIR.joinpath("dataset/train_dataset_image_mean.png"),
            interactive=interactive_reports,
        )

        val_x_batch_shape = train_ds.get_x_batch_shape()
        print("\nValidation dataset's images batch shape is:")
        print(val_x_batch_shape)

        val_y_batch_shape = train_ds.get_y_batch_shape()
        print("\nValidation dataset's labels batch shape is:")
        print(val_y_batch_shape)

        val_ds.display_images_in_batch(
            2,
            "Validation dataset",
            CHARTS_DIR.joinpath("dataset/test_dataset_image_sample.png"),
            interactive=interactive_reports,
        )
        val_ds.display_batch_number(
            "Validation dataset",
            CHARTS_DIR.joinpath("dataset/test_dataset_batch_number.png"),
            interactive=interactive_reports,
        )
        val_ds.display_distribution(
            "Validation dataset",
            CHARTS_DIR.joinpath("dataset/test_dataset_image_distribution.png"),
            interactive=interactive_reports,
        )
        val_ds.display_mean(
            "Validation dataset",
            CHARTS_DIR.joinpath("dataset/test_dataset_image_mean.png"),
            interactive=interactive_reports,
        )

        self.batch_size = batch_size
        self.class_names = class_names
        self.img_color = img_color
        self.img_size = img_size
        self.interactive_reports = interactive_reports
        self.train_ds = train_ds.normalized_dataset
        self.val_ds = val_ds.normalized_dataset

    def build(self):
        channels = 1 if self.img_color == "grayscale" else 3

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(
                    input_shape=(self.img_size, self.img_size, channels)
                ),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        model.summary()

        return model

    def train(self, epochs):
        model = self.build()

        print("\nStarting training...")

        history = model.fit(
            self.train_ds,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=self.val_ds,
        )

        print("\n\033[92mTraining done !\033[0m")

        plot_history(
            history,
            CHARTS_DIR.joinpath("training_metrics/training_loss_and_accuracy.png"),
            accuracy_metric="categorical_accuracy",
            interactive=self.interactive_reports,
        )

        print("\nSaving model...")

        model.save(MODELS_DIR.joinpath("model_1.keras"))

        print("\n\033[92mSaving done !\033[0m")
