import tensorflow as tf
import tensorflowjs as tfjs
import pathlib
import numpy as np

from sklearn.model_selection import KFold
from sklearn.utils import class_weight

from x_ray_dataset_builder import Dataset    


class Model:
    def __init__(self, image_size=(180, 180)):
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
        self.model = None
        self.train_ds = train_ds.normalized_dataset
        self.x_train = train_ds.x_dataset
        self.y_train = train_ds.y_dataset

    
    def build(self):

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.RandomZoom(0.2, input_shape=(180, 180, 1)))
        model.add(tf.keras.layers.RandomRotation(0.1))
        model.add(tf.keras.layers.RandomContrast(0.1))

        model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.7))

        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(2, activation="softmax"))

        optimizer_func = tf.keras.optimizers.experimental.Adagrad(learning_rate=0.005)

        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        model.compile(optimizer=optimizer_func, loss=loss_func, metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
            ])
        
        self.model = model

        return model


    def train(self, epochs, k=5):
        k = k
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        fold = 1

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=np.argmax(self.y_train, axis=1))
        class_weights = dict(enumerate(class_weights))
        class_weights[0] = class_weights[0] * 12.25

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', mode='max', patience=10, restore_best_weights=True)


        for train_index, val_index in kfold.split(self.x_train, self.y_train):       
            model = self.build()

            print(f"\nProcessing fold {fold}")
            train_images, val_images = self.x_train[train_index], self.x_train[val_index]
            train_labels, val_labels = self.y_train[train_index], self.y_train[val_index]

            model.fit(
                train_images,
                train_labels,
                class_weight=class_weights,
                batch_size=32,
                epochs=epochs,
                validation_data=(val_images, val_labels),
                callbacks=[stop_early]
            )
            
            fold += 1
        
        print("\n\033[92mTraining done !\033[0m")

        print("\nSaving...")
        model.save("notebooks/6_data_augmentation/model_6.keras")
        tfjs.converters.save_keras_model(model, "notebooks/6_data_augmentation")
        print("\n\033[92mSaving done !\033[0m")