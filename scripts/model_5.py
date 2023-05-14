import tensorflow as tf
import tensorflowjs as tfjs
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt

from sklearn.model_selection import KFold
from sklearn.utils import class_weight

from dataset import Dataset


def model_builder(hp):
    model = tf.keras.Sequential(
        [
            tf.keras.ayers.RandomZoom(0.2, input_shape=(256, 256, 1)),
            tf.keras.ayers.RandomRotation(0.1),
            tf.keras.ayers.RandomContrast(0.1),
        ]
    )

    hp_l2_reg = hp.Float('l2_regularizer', min_value=1e-5, max_value=1e-2, sampling="log")
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

    # rescale=1./255,        # normalize pixel values
    # width_shift_range=0.1, # randomly shift images horizontally (10% of total width)
    # height_shift_range=0.1,# randomly shift images vertically (10% of total height)
    # zoom_range=0.2,        # randomly zoom images in or out by 20%
    # fill_mode='nearest'    # set mode for filling points outside the input boundaries

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp_l2_reg), input_shape=(256, 256, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp_l2_reg)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp_l2_reg)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp_l2_reg)))
    model.add(tf.keras.layers.Dropout(hp_dropout_rate))  # Add dropout after the Dense layer

    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling="log")

    optimizer_func = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=[
            tf.keras.metrics.CategoricalAccuracy(), 
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall(),
            tf.keras.metrics.F1Score(),
            tf.keras.metrics.AUC()
        ])

    return model



class HyperModel:
    def __init__(self, x_train, y_train, max_epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.max_epochs = max_epochs
        self.hypermodel = None

    def build(self):
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        hyper_band = kt.Hyperband(model_builder, objective=kt.Objective('val_recall', direction='max'), max_epochs=50, factor=3, directory='hypertunning_logs', project_name='hyperband_algo_5')
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=np.argmax(self.y_train, axis=1))
        class_weights = dict(enumerate(class_weights))
        class_weights[0] = class_weights[0] * 4
        hyper_band.search(self.x_train, self.y_train, class_weight=class_weights, validation_split=0.20, callbacks=[stop_early], epochs=self.max_epochs, batch_size=32)
        print("\n")
        hyper_band.results_summary(1)
        best_hyperparameters = hyper_band.get_best_hyperparameters()[0]
        hypermodel = hyper_band.hypermodel.build(best_hyperparameters)
        self.hypermodel = hypermodel

        return hypermodel


class Model:
    def __init__(self, image_size=(512, 512)):
        train_dir = pathlib.Path("data/train")

        train_ds = Dataset(train_dir, batch_size=32, image_size=image_size)

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

        train_ds.display_images_in_batch(1, "Training dataset")
        train_ds.display_batch_number("Training dataset")
        train_ds.display_distribution("Training dataset")
        train_ds.display_mean("Training dataset")

        self.class_names = class_names
        self.train_ds = train_ds.normalized_dataset
        self.x_train = train_ds.x_dataset
        self.y_train = train_ds.y_dataset


    def train(self, epochs, max_epochs, k=5):
        hypermodel = HyperModel(self.x_train, self.y_train, max_epochs)

        k = k
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        fold = 1

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=np.argmax(self.y_train, axis=1))
        class_weights = dict(enumerate(class_weights))
        class_weights[0] = class_weights[0] * 6.75


        for train_index, val_index in kfold.split(self.x_train, self.y_train):       
            model = hypermodel.build()

            print(f"\nProcessing fold {fold}")
            train_images, val_images = self.x_train[train_index], self.x_train[val_index]
            train_labels, val_labels = self.y_train[train_index], self.y_train[val_index]

            history = model.fit(train_images, train_labels, class_weight=class_weights, batch_size=32, epochs=epochs, validation_data=(val_images, val_labels))
            
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
        model.save("notebooks/5_regularization/model_5.keras")
        tfjs.converters.save_keras_model(model, "notebooks/5_regularization")
        print("\n\033[92mSaving done !\033[0m")