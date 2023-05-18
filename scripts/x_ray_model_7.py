import numpy as np
import pathlib
import tensorflow as tf
import tensorflowjs as tfjs

from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from custom_layer import ConcatenationLayer
from x_ray_dataset_builder import Dataset
from x_ray_data_viz import plot_history


class Model:
    def __init__(self, batch_size=32, image_size=(256, 256)):
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
        self.scores = None
        self.acc_per_fold = []
        self.loss_per_fold = []

    def build(self):
        input_shape = (256, 256, 3)

        img_input = tf.keras.layers.Input(shape=input_shape)

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

        outputs = tf.keras.layers.BatchNormalization()(concat_layer)
        outputs = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(outputs)
        outputs = tf.keras.layers.Dropout(0.5)(outputs)

        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(outputs)
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

        history = model.fit(self.train_ds, epochs=25, validation_data=self.test_ds)

        plot_history(history=history)

        return model

    def train(self, epochs, k=5):
        class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(self.y_train),
            y=(self.y_train> 0.5).astype("int32").reshape(-1),
        )
        class_weights = dict(enumerate(class_weights))
        # {0: 1.8831227436823104, 1: 0.6807504078303426}

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
            cooldown=4,
            factor=0.001,
            min_delta=0.01,
            monitor="val_binary_accuracy",
            patience=8,
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

        x_all = np.concatenate((self.x_train, self.x_test), axis=0)
        y_all = np.concatenate((self.y_train, self.y_test), axis=0)
        kfold = KFold(n_splits=k, shuffle=True)
        fold_no = 1

        for train, test in kfold.split(x_all, y_all):
            print("\n\033[91m=================================================================\033[0m")
            print(f"\033[91m****************************TRAINING FOLD N°{fold_no}**************************\033[0m")
            print("\033[91m===================================================================\033[0m\n")

            model = self.build()
            model.trainable = True
            model.summary()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, amsgrad=True),
                loss='binary_crossentropy',
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
            history = model.fit(
                x_all[train],
                y_all[train],            
                batch_size=self.batch_size,
                callbacks=[stop_early, reduce_learning_rate, model_save],
                class_weight=class_weights,
                epochs=epochs,
            )
            self.scores = model.evaluate(x_all[test], y_all[test], verbose=1)

            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {self.scores[0]}; {model.metrics_names[1]} of {self.scores[1]*100}%')

            self.acc_per_fold.append(self.scores[1] * 100)
            self.loss_per_fold.append(self.scores[0])

            print("\nSaving...")

            model.save(f"notebooks/7_transfer_learning/model_7_fold_{fold_no}.keras")
            tfjs.converters.save_keras_model(model, "notebooks/7_transfer_learning")

            print("\n\033[92mSaving done !\033[0m")

            plot_history(history=history)
            fold_no = fold_no + 1

        print('Score per fold')

        for i in range(0, len(self.acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {self.loss_per_fold[i]} - Accuracy: {self.acc_per_fold[i]}%')
        
        print("\n\033[91m=================================================================\033[0m")
        print(f"\033[91m********************AVERAGE SCORES FOR ALL FOLDS******************\033[0m")
        print("\033[91m===================================================================\033[0m\n")        
        print(f'> Accuracy: {np.mean(self.acc_per_fold)} (+- {np.std(self.acc_per_fold)})')
        print(f'> Loss: {np.mean(self.loss_per_fold)}')
        print("\n\033[91m=================================================================\033[0m")
        print(f"\033[91m****************************TRAINING DONE**************************\033[0m")
        print("\033[91m===================================================================\033[0m\n")
        print("\n\033[92mTraining done !\033[0m")
