import pathlib
import tensorflow as tf
import numpy as np

from scripts.data_vizualisation import plot_image, plot_distribution, plot_mean


class BasicModel(tf.keras.Sequential):
    def __init__(self):
        # Load datasets
        train_dir = pathlib.Path("data/chest_xray/train")

        test_dir = pathlib.Path("data/chest_xray/test")


        training_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            color_mode="grayscale",
            batch_size=None,
            image_size=(256, 256),
            shuffle=True,
        )

        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            color_mode="grayscale",
            batch_size=None,
            image_size=(256, 256),
            shuffle=True,
        )


        training_dataset = training_dataset.batch(32, drop_remainder=True)

        test_dataset = test_dataset.batch(32, drop_remainder=True)


        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

        normalized_train_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))

        normalized_test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

        # Initialize lists for training and testing data
        x_train, y_train = [], []
        x_test, y_test = [], []

        # Retrieve training data
        for x, y in normalized_train_dataset.unbatch().as_numpy_iterator():
            x_train.append(x)
            y_train.append(y[0])

        # Retrieve testing data
        for x, y in normalized_test_dataset.unbatch().as_numpy_iterator():
            x_test.append(x)
            y_test.append(y[0])

        # Convert lists to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        """
        Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
        To do so, divide the values by 255. 
        It's important that the training set and the testing set be preprocessed in the same way.
        """
        
        # x_train, x_test = x_train / 255.0, x_test / 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_image(self):
        # Check image content by displaying one digit image from an index input
        plot_image(self.x_train, self.y_train)

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
            tf.keras.layers.Flatten(input_shape=(256, 256)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
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

        return model

    def train(self, epochs, x_val, y_val):
        """
        Instanciate the hypermodel with current training and testing datasets.
        Also pass the limit of epochs that could will run in order to find the optimal number of epoch
        """

        # Build the model and get or set the search
        model = self.build()

        # First trainign test to find the optimal number of epoch
        epochs = epochs

        # Train the model
        print('\nStarting training...')
        # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),validation_steps=32, validation_freq=1, validation_batch_size=32, epochs=epochs)
        print('\n\033[92mTraining done !\033[0m')

        print('\nEvaluating model...')
        print('\n')
        print(f'\n{x_val}')
        print(f'\n{y_val}')
        test_loss, test_acc = model.evaluate(x=x_val, y=y_val, verbose=2)
        print('\nTest loss is: %s' % (test_loss))
        print('\nTest accurancy is: %s' % (test_acc))

        # Save the model so he could be infer an unlimited amount of time without training again
        print('\nSaving...')
        model.save("notebooks/1_train_validation_test_procedure/model_1.h5")
        print('\n\033[92mSaving done !\033[0m')