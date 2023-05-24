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


class MergeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MergeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.layers.Concatenate()(inputs)


class Model:
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        img_color: str,
        label_mode: str,
        interactive_reports: bool = True,
    ):
        train_path = pathlib.Path("data/train")

        test_dir = pathlib.Path("data/test")

        train_ds = Dataset(
            train_path,
            batch_size=batch_size,
            color_mode=img_color,
            image_size=img_size,
            label_mode=label_mode,
        )

        test_ds = Dataset(
            test_dir,
            image_size=img_size,
            batch_size=1,
            color_mode=img_color,
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
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )

        callback_reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
            cooldown=2,
            factor=0.1,
            min_delta=0.01,
            mode="max",
            monitor="binary_accuracy",
            patience=4,
            verbose=1,
        )

        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR.joinpath(
                f"checkpoints/model_7_checkpoint_{datetime.utcnow().isoformat()}.keras"
            ),
            mode="max",
            monitor="val_binary_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

        self.batch_size = batch_size
        self.callback_stop_early = callback_stop_early
        self.callback_reduce_learning_rate = callback_reduce_learning_rate
        self.callback_checkpoint = callback_checkpoint
        self.class_names = train_ds.class_names
        self.class_weights = class_weights
        self.fold_acc = []
        self.fold_loss = []
        self.img_size = img_size
        self.img_color = img_color
        self.interactive_reports = interactive_reports
        self.model = None
        self.scores = []
        self.x_train = train_ds.x_dataset
        self.x_test = test_ds.x_dataset
        self.y_train = train_ds.y_dataset
        self.y_test = test_ds.y_dataset

    def build(self):
        input_shape = (self.img_size, self.img_size, 3)

        input_layer = tf.keras.layers.Input(shape=input_shape)

        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )

        inception_v3 = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )

        resnet50 = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )

        for layer in vgg16.layers:
            layer.trainable = False
        for layer in inception_v3.layers:
            layer.trainable = False
        for layer in resnet50.layers:
            layer.trainable = False

        channels = 1 if self.img_color == "grayscale" else 3

        augmentation_layer = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(
                    0.3, input_shape=(IMG_SIZE, IMG_SIZE, channels)
                ),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
            ]
        )

        augmented_input_layer = augmentation_layer(input_layer)

        vgg16_outputs = vgg16(augmented_input_layer)
        vgg16_outputs = tf.keras.layers.GlobalAveragePooling2D()(vgg16_outputs)

        inception_v3_outputs = inception_v3(augmented_input_layer)
        inception_v3_outputs = tf.keras.layers.GlobalAveragePooling2D()(
            inception_v3_outputs
        )

        resnet50_outputs = resnet50(augmented_input_layer)
        resnet50_outputs = tf.keras.layers.GlobalAveragePooling2D()(resnet50_outputs)

        trained_models_layer = MergeLayer()(
            [vgg16_outputs, inception_v3_outputs, resnet50_outputs]
        )

        model_outputs = tf.keras.layers.BatchNormalization()(trained_models_layer)
        model_outputs = tf.keras.layers.Dense(
            256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(model_outputs)
        model_outputs = tf.keras.layers.Dropout(0.5)(model_outputs)

        model_outputs = tf.keras.layers.BatchNormalization()(model_outputs)
        model_outputs = tf.keras.layers.Dense(
            128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(model_outputs)
        model_outputs = tf.keras.layers.Dropout(0.7)(model_outputs)

        model_outputs = tf.keras.layers.Dense(1, activation="sigmoid")(model_outputs)

        model = tf.keras.Model(inputs=input_layer, outputs=model_outputs)

        model.compile(
            loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
            optimizer=tf.keras.optimizers.AdamW(
                amsgrad=True,
                epsilon=0.0001,
                learning_rate=0.004,
            ),
        )

        model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            callbacks=[self.callback_reduce_learning_rate, self.callback_stop_early],
            class_weight=self.class_weights,
            epochs=self.batch_size,
            shuffle=True,
            validation_split=0.2,
        )

        model.evaluate(
            self.x_test,
            self.y_test,
            batch_size=1,
            verbose=1,
        )

        self.model = model

        return model

    def train(self, epochs, k):
        kfold = KFold(n_splits=k, shuffle=True)
        fold_number = 1

        for train, test in kfold.split(self.x_train, self.y_train):
            print(
                "\033[91m"
                "=================================================================\n"
                "****STARTING RESNET_50, VGG_16 and INCEPTION_V3 PRE-TRAINING*****\n"
                "================================================================="
                "\033[0m"
            )

            train_images, val_images = self.x_train[train], self.x_train[test]
            train_labels, val_labels = self.y_train[train], self.y_train[test]

            print("\033[96mBuilding model...\n")

            pretrained_model = self.build()

            print("\033[0m")

            print(
                "\n\033[91m"
                "=================================================================\n"
                "****PRE-TRAINING DONE FOR RESNET_50, VGG_16 and INCEPTION_V3*****\n"
                "================================================================="
                "\033[0m\n"
            )

            print(
                "\033[91m"
                "=================================================================\n"
                f"**************STARTING TRAINING K-FOLD N°{fold_number}***********\n"
                "================================================================="
                "\033[0m"
            )

            for layer in pretrained_model.layers:
                layer.trainable = True

            print("\n\033[96mModel summary:\033[0m\n")

            pretrained_model.summary()

            model = pretrained_model

            print("\n\033[96m")

            model.compile(
                loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
                optimizer=tf.keras.optimizers.AdamW(
                    amsgrad=True,
                    epsilon=0.01,
                    learning_rate=0.0004,
                ),
            )

            history = model.fit(
                train_images,
                train_labels,
                batch_size=self.batch_size,
                callbacks=[
                    self.callback_stop_early,
                    self.callback_reduce_learning_rate,
                    self.callback_checkpoint,
                ],
                class_weight=self.class_weights,
                epochs=epochs,
                shuffle=True,
                validation_data=(val_images, val_labels),
            )

            self.scores = model.evaluate(
                self.x_test, self.y_test, batch_size=1, verbose=1
            )

            self.fold_acc.append(self.scores[1] * 100)
            self.fold_loss.append(self.scores[0])

            print("\033[0m")

            print(
                "\n\033[91m"
                "=================================================================\n"
                f"**************TRAINING FOR K-FOLD N°{fold_number} DONE!**********\n"
                "================================================================="
                "\n\033[0m"
            )

            print(f"\n\033[91mSaving model n°{fold_number}...\033[0m")
            
            self.model.save(MODEL_DIR.joinpath(f"model_7_fold_{fold_number}.keras"))

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
            f"""
            \n\033[92m
            =================================================================\\
            ********************AVERAGE SCORES FOR ALL FOLDS*****************\\
            =================================================================
            \033[0m\n
            """
        )
        print(
            f"> Average accuracy: {np.mean(self.fold_acc)} (+- {np.std(self.fold_acc)})"
        )
        print(f"> Average loss: {np.mean(self.fold_loss)}")
        print(
            """\n\033[91m=================================================================
            ****************************TRAINING DONE************************
            =================================================================\033[0m\n"""
        )
