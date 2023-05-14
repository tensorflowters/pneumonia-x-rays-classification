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

        train_ds = Dataset(train_dir, batch_size=32, image_size=image_size, color_mode='rgb', validation_split=0.2, subset='training')
        test_ds = Dataset(train_dir, batch_size=32, image_size=image_size, color_mode='rgb', validation_split=0.2, subset='training')

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds.build(AUTOTUNE, False)
        test_ds.build(AUTOTUNE, False)

        self.class_names = train_ds.class_names
        self.model = None
        self.base_model = None
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.x_train = train_ds.x_dataset
        self.raw_x_dataset = train_ds.raw_x_dataset
        self.y_train = train_ds.y_dataset

    
    def init_build(self):
        data_augmentation = tf.keras.Sequential(
            [tf.keras.layers.RandomZoom(0.2, input_shape=(180, 180, 3)), tf.keras.layers.RandomRotation(0.1), tf.keras.layers.RandomContrast(0.1),]
        )
        # Load the EfficientNet model with pre-trained ImageNet weights
        base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

        # Freeze the base model (so its weights won't change during training)
        base_model.trainable = False

        self.base_model = base_model
        inputs = tf.keras.Input(shape=(180, 180, 3))
        x = data_augmentation(inputs)  # Apply random data augmentation
        # Pre-trained Xception weights requires that input be scaled
        # from (0, 255) to a range of (-1., +1.), the rescaling layer
        # outputs: `(inputs * scale) + offset`
        scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
        x = scale_layer(x)

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        # optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
            ],
        )

        epochs = 20
        # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=np.argmax(self.y_train, axis=1))
        # class_weights = dict(enumerate(class_weights))

        model.fit(self.train_ds.raw_dataset, epochs=epochs, validation_data=self.test_ds.raw_dataset)
        
        self.model = model

        return model


    def train(self, epochs, k=5):
        k = k
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        fold = 1

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=np.argmax(self.y_train, axis=1))
        class_weights = dict(enumerate(class_weights))
        # class_weights[0] = class_weights[0] * 12.75

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', mode='max', patience=20, restore_best_weights=True)

        model = self.init_build()

        self.base_model.trainable = True

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
            ],
        )

        epochs = 100

        for train_index, val_index in kfold.split(self.raw_x_dataset, self.y_train):       

            print(f"\nProcessing fold {fold}")
            train_images, val_images = self.raw_x_dataset[train_index], self.raw_x_dataset[val_index]
            train_labels, val_labels = self.y_train[train_index], self.y_train[val_index]

            model.fit(train_images, train_labels, class_weight=class_weights, batch_size=32, epochs=epochs, validation_data=(val_images, val_labels), callbacks=[stop_early])
            
            fold += 1
        
        print("\n\033[92mTraining done !\033[0m")

        print("\nSaving...")
        model.save("notebooks/7_transfer_learning/model_7.keras")
        tfjs.converters.save_keras_model(model, "notebooks/7_transfer_learning")
        print("\n\033[92mSaving done !\033[0m")