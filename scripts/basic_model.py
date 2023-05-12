import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from scripts.data_vizualisation import plot_image, plot_distribution, plot_mean


class BasicModel:
    def __init__(self):
        train_dir = pathlib.Path("data/chest_xray/train")
        test_dir = pathlib.Path("data/chest_xray/test")


        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            # labels="inferred",
            # label_mode="binary",
            validation_split=0.2,
            subset="training",
            seed=123,
            # color_mode="grayscale",
            batch_size=32,
            image_size=(180, 180),
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            # labels="inferred",
            # label_mode="binary",
            validation_split=0.2,
            subset="validation",
            seed=123,
            # color_mode="grayscale",
            batch_size=32,
            image_size=(180, 180),
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            # labels="inferred",
            # label_mode="binary",
            # color_mode="grayscale",
            seed=123,
            batch_size=32,
            image_size=(180, 180),
        )

        class_names = train_ds.class_names
        print(class_names)

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        
        self.train_ds = train_ds
        # self.train_dataset = normalized_train_dataset
        # self.x_train = x_train
        # self.y_train = y_train

        self.val_ds = val_ds
        # self.val_ds = normalized_validation_dataset
        # self.x_val = x_val
        # self.y_val = y_val

        self.test_ds = test_ds
        # self.test_ds = normalized_test_dataset
        # self.x_test = x_test
        # self.y_test = y_test

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
        The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 256 by 256 pixels)
        to a one-dimensional array (of 256 * 256 = 65 536 pixels).

        Think of this layer as unstacking rows of pixels in the image and lining them up.

        This layer has no parameters to learn; it only reformats the data.

        After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.

        These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons).

        The second (and last) layer returns a logits array with length of 2.

        Each node contains a score that indicates the current image belongs to one of the 2 classes.

    Args:
        None
    """
    def build(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
            # tf.keras.layers.Dense(num_classes)
        ])

        """
        Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
            - Loss function — This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
            - Optimizer — This is how the model is updated based on the data it sees and its loss function.
            - Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
        """
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
        # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
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