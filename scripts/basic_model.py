import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_vizualisation import plot_distribution, plot_mean


class BasicModel:
    def __init__(self):
        train_dir = pathlib.Path("data/chest_xray/train")
        test_dir = pathlib.Path("data/chest_xray/test")


        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            validation_split=0.2,
            subset="training",
            seed=123,
            color_mode="grayscale",
            batch_size=32,
            image_size=(128, 128),
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            validation_split=0.2,
            subset="validation",
            seed=123,
            color_mode="grayscale",
            batch_size=32,
            image_size=(128, 128),
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            color_mode="grayscale",
            seed=123,
            batch_size=32,
            image_size=(128, 128),
        )

        class_names = train_ds.class_names
        self.class_names = class_names
        print(class_names)

        for image_batch, labels_batch in train_ds:
            print(image_batch)
            print(image_batch.shape)
            print(labels_batch)
            print(labels_batch.shape)
            break

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        self.train_ds = train_ds

        self.val_ds = val_ds

        self.test_ds = test_ds

        self.class_names = class_names


    def get_image(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.show()

    def get_distribution(self):
        # Display digit distribution in training and testing datasets
        plot_distribution(self.y_train, self.y_test)

    def get_mean(self):
        # Display digit mean occurence in training and testing datasets
        plot_mean(self.y_train, self.y_test)

    """
    Function:

    Args:
    
    """
    def build(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        """
        Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
            - Loss function — This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
            - Optimizer — This is how the model is updated based on the data it sees and its loss function.
            - Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
        """
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        model.summary()

        return model

    def train(self, epochs):
        """
        
        """

        model = self.build()

        
        # Train the model
        print('\nStarting training...')
        history = model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs)
        print('\n\033[92mTraining done !\033[0m')

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        # Save the model so he could be infer an unlimited amount of time without training again
        print('\nSaving...')
        model.save("notebooks/1_train_validation_test_procedure/model_1.h5")
        print('\n\033[92mSaving done !\033[0m')

    
    def evaluate(self):
        model = tf.keras.models.load_model("notebooks/1_train_validation_test_procedure/model_1.h5")
        print('\nEvaluating model...')
        test_loss, test_acc = model.evaluate(self.test_ds)
        print('\nTest loss is: %s' % (test_loss)) 
        print('\nTest accurancy is: %s' % (test_acc))