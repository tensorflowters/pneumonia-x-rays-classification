import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import Dataset


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

        """
        Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
            - Loss function — This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
            - Optimizer — This is how the model is updated based on the data it sees and its loss function.
            - Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
        """
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        model.summary()

        return model

    def train(self, epochs):
        model = self.build()

        # Train the model
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

        # Save the model so he could be infer an unlimited amount of time without training again
        print("\nSaving...")
        model.save("notebooks/1_train_validation_test_procedure/model_1.h5")
        print("\n\033[92mSaving done !\033[0m")
