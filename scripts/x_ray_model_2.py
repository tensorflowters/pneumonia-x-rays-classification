import pathlib
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from scripts.x_ray_dataset_builder import Dataset


class Model:
    def __init__(self, image_size=(512, 512)):
        train_dir = pathlib.Path("data/train")

        train_ds = Dataset(train_dir, batch_size=64, image_size=image_size)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds.build(AUTOTUNE, False)

        class_names = train_ds.get_class_names()
        print("\nClass names:")
        print(class_names)

        train_x_batch_shape = train_ds.get_x_batch_shape()
        print("\nTraining dataset's images batch shape is:")
        print(train_x_batch_shape)

        train_y_batch_shape = train_ds.get_y_batch_shape()
        print("\nTraining dataset's labels batch shape is:")
        print(train_y_batch_shape)

        train_ds.display_images_in_batch(2, "Training dataset")
        train_ds.display_batch_number("Training dataset")
        train_ds.display_distribution("Training dataset")
        train_ds.display_mean("Training dataset")

        self.class_names = class_names
        self.train_ds = train_ds.normalized_dataset
        self.x_train = train_ds.x_dataset
        self.y_train = train_ds.y_dataset

    def build(self, input_shape=(512, 512, 1)):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(len(self.class_names), activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(), 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()
            ],
        )

        model.summary()

        return model
    
    def train(self, epochs, k=5, input_shape=(512, 512, 1)):
        print("\nStarting training...")
        k = k
        num_epochs = epochs

        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        fold = 1

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=np.argmax(self.y_train, axis=1))
        class_weights = dict(enumerate(class_weights))
        class_weights[0] = class_weights[0] * 4.25


        for train_index, val_index in kfold.split(self.x_train, self.y_train):       
            model = self.build(input_shape=input_shape)

            print(f"Processing fold {fold}")
            train_images, val_images = self.x_train[train_index], self.x_train[val_index]
            train_labels, val_labels = self.y_train[train_index], self.y_train[val_index]

            history = model.fit(train_images, train_labels, class_weight=class_weights, batch_size=64, epochs=num_epochs, validation_data=(val_images, val_labels))
            
            fold += 1

            categorical_accuracy = history.history["categorical_accuracy"]
            val_categorical_accuracy = history.history["val_categorical_accuracy"]

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]

            epochs_range = range(epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, categorical_accuracy, label="Training Accuracy")
            plt.plot(epochs_range, val_categorical_accuracy, label="Validation Accuracy")
            plt.legend(loc="lower right")
            plt.title("Training and Validation Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label="Training Loss")
            plt.plot(epochs_range, val_loss, label="Validation Loss")
            plt.legend(loc="upper right")
            plt.title("Training and Validation Loss")
            plt.show()
        
        print("\n\033[92mTraining done !\033[0m")

        print("\nSaving...")
        model.save("notebooks/2_cross_validation/model_2.keras")
        model.save("notebooks/2_cross_validation")
        print("\n\033[92mSaving done !\033[0m")