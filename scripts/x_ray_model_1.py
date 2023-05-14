import pathlib
import tensorflow as tf
import tensorflowjs as tfjs
import matplotlib.pyplot as plt

from scripts.x_ray_dataset_builder import Dataset


class Model:
    def __init__(self):
        train_dir = pathlib.Path("data/train")

        train_ds = Dataset(train_dir, 0.2, "training")
        val_ds = Dataset(train_dir, 0.2, "validation")

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

        train_ds.display_images_in_batch(2, "Training dataset")
        train_ds.display_batch_number("Training dataset")
        train_ds.display_distribution("Training dataset")
        train_ds.display_mean("Training dataset")

        val_x_batch_shape = train_ds.get_x_batch_shape()
        print("\nTesting dataset's images batch shape is:")
        print(val_x_batch_shape)

        val_y_batch_shape = train_ds.get_y_batch_shape()
        print("\nTesting dataset's labels batch shape is:")
        print(val_y_batch_shape)

        val_ds.display_images_in_batch(2, "Testing dataset")
        val_ds.display_batch_number("Testing dataset")
        val_ds.display_distribution("Testing dataset")
        val_ds.display_mean("Testing dataset")

        self.class_names = class_names
        self.train_ds = train_ds.normalized_dataset
        self.val_ds = val_ds.normalized_dataset

    def build(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(512, 512, 1)),
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

    def train(self, epochs):
        model = self.build()

        print("\nStarting training...")
        history = model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs)
        print("\n\033[92mTraining done !\033[0m")

        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

        print("\nSaving...")
        model.save("notebooks/1_train_validation_test_procedure/model_1.keras")
        tfjs.converters.save_keras_model(model, "notebooks/1_train_validation_test_procedure")
        print("\n\033[92mSaving done !\033[0m")
