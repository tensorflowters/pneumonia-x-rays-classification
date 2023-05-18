import numpy as np
import pathlib
import tensorflow as tf
import tensorflowjs as tfjs

from sklearn.utils import class_weight

from custom_layer import ConcatenationLayer
from x_ray_dataset_builder import Dataset


class Model:
    def __init__(self, batch_size=16, image_size=(180, 180)):
        train_dir = pathlib.Path("data/train")

        train_ds = Dataset(
            train_dir,
            batch_size=batch_size,
            color_mode="rgb",
            image_size=image_size,
            label_mode="binary",
            subset="training",
            validation_split=0.2,
        )

        test_ds = Dataset(
            train_dir,
            batch_size=batch_size,
            color_mode="rgb",
            image_size=image_size,
            label_mode="binary",
            subset="validation",
            validation_split=0.2,
        )

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds.build(AUTOTUNE, True)
        test_ds.build(AUTOTUNE, False)

        self.batch_size = batch_size
        self.class_names = train_ds.class_names
        self.model = None
        self.test_ds = test_ds.dataset
        self.train_ds = train_ds.dataset
        self.x_test = test_ds.x_dataset
        self.x_train = train_ds.x_dataset
        self.y_test = test_ds.y_dataset
        self.y_train = train_ds.y_dataset

    def build(self):
        input_shape = (180, 180, 3)

        resnet_base = tf.keras.applications.resnet50.ResNet50(
            weights="imagenet", input_shape=input_shape, include_top=False
        )

        vgg_base = tf.keras.applications.vgg16.VGG16(
            weights="imagenet", input_shape=input_shape, include_top=False
        )

        inception_base = tf.keras.applications.inception_v3.InceptionV3(
            weights="imagenet", input_shape=input_shape, include_top=False
        )

        resnet_base.trainable = False
        vgg_base.trainable = False
        inception_base.trainable = False

        img_input = tf.keras.layers.Input(shape=input_shape)

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", input_shape=input_shape),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2),
            ]
        )

        augmented_inputs = data_augmentation(img_input)

        resnet_output = resnet_base(augmented_inputs)
        vgg_output = vgg_base(augmented_inputs)
        inception_output = inception_base(augmented_inputs)

        resnet_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)
        vgg_output = tf.keras.layers.GlobalAveragePooling2D()(vgg_output)
        inception_output = tf.keras.layers.GlobalAveragePooling2D()(inception_output)

        concat_layer = ConcatenationLayer()(
            [
                resnet_output, 
                vgg_output, 
                inception_output
            ]
        )

        model = tf.keras.layers.BatchNormalization()(concat_layer)
        outputs = tf.keras.layers.Dense(256, activation="relu")(model)
        outputs = tf.keras.layers.Dropout(0.5)(outputs)

        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Dense(128, activation="relu")(outputs)
        outputs = tf.keras.layers.Dropout(0.5)(outputs)

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(outputs)

        model = tf.keras.Model(inputs=img_input, outputs=outputs)

        self.model = model

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        model.fit(self.train_ds, epochs=10, validation_data=self.test_ds)

        return model

    def train(self, epochs):
        class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(self.y_train),
            y=np.argmax(self.y_train, axis=1),
        )
        class_weights = dict(enumerate(class_weights))
        # class_weights[0] = class_weights[0] * 12.5

        stop_early = tf.keras.callbacks.EarlyStopping(
            mode="max",
            monitor="val_binary_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.01,
            min_delta=0.01,
            mode="min",
            monitor="val_loss",
            patience=5,
            verbose=1,
        )

        model_save = tf.keras.callbacks.ModelCheckpoint(
            "notebooks/7_transfer_learning/model_7_checkpoint.keras",
            mode="min",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

        model = self.build()

        model.trainable = True

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        model.fit(
            self.train_ds,
            batch_size=self.batch_size,
            callbacks=[stop_early, reduce_learning_rate, model_save],
            class_weight=class_weights,
            epochs=epochs,
            steps_per_epoch=int(len(self.x_train)/self.batch_size),
            validation_data=(self.test_ds),
        )

        print("\n\033[92mTraining done !\033[0m")

        print("\nSaving...")

        model.save("notebooks/7_transfer_learning/model_7.keras")
        tfjs.converters.save_keras_model(model, "notebooks/7_transfer_learning")

        print("\n\033[92mSaving done !\033[0m")
