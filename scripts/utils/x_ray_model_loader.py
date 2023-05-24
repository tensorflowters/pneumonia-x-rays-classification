import os
import math
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report

from utils.x_ray_data_viz import plot_confusion_matrix, plot_roc_curve
from utils.x_ray_dataset_builder import Dataset


class ModelLoader:
    def __init__(
        self,
        batch_size=40,
        color="grayscale",
        img_size=256,
        label_mode="binary",
        path_to_register_charts: str = None,
        interactive_reports: bool = True,
    ):
        pred_list = os.listdir("data/prediction/")

        test_dir = pathlib.Path("data/test")

        test_ds = Dataset(
            test_dir,
            batch_size=batch_size,
            color_mode=color,
            image_size=img_size,
            label_mode=label_mode,
        )

        AUTOTUNE = tf.data.AUTOTUNE

        test_ds.build(AUTOTUNE)

        self.class_names = test_ds.get_class_names()
        self.interactive_reports = interactive_reports
        self.label_mode = label_mode
        self.loaded_model = None
        self.path_to_register_charts = path_to_register_charts
        self.pred_list = pred_list
        self.probability_model = None
        self.test_ds = test_ds.normalized_dataset
        self.x_test = test_ds.x_dataset
        self.y_test = test_ds.y_dataset

    def load(self, model_pathname, **kwargs):
        print("\n\033[94mModel loading...\033[0m")

        self.loaded_model = tf.keras.models.load_model(model_pathname, **kwargs)

        self.probability_model = tf.keras.Sequential(
            [self.loaded_model, tf.keras.layers.Softmax()]
        )

        print("\n\033[92mModel successfully loaded!\033[0m")

    def evaluate(self, batch_size: int = 32, binary=False):
        print("\n\033[94mEvaluating model...\033[0m\n")

        if binary:
            (
                test_loss,
                test_binary_accuracy,
                test_precision,
                test_recall,
            ) = self.loaded_model.evaluate(self.x_test, self.y_test, verbose=2)

            print("\n\033[94mEvaluation loss is: %s\033[0m" % (test_loss))
            print(
                "\n\033[94mEvaluation binary accurancy is: %s\033[0m"
                % (test_binary_accuracy)
            )
            print("\n\033[94mEvaluation precision is: %s\033[0m" % (test_precision))
            print("\n\033[94mEvaluation recall is: %s\n\033[0m" % (test_recall))
        else:
            (
                test_loss,
                binary_accuracy,
                test_precision,
                test_recall,
            ) = self.loaded_model.evaluate(self.x_test, self.y_test, verbose=2)
            print("\n\033[94mEvaluation loss is: %s\033[0m" % (test_loss))
            print(
                "\n\033[94mEvaluation binary accurancy is: %s\033[0m"
                % (binary_accuracy)
            )
            print("\n\033[94mEvaluation precision is: %s\033[0m" % (test_precision))
            print("\n\033[94mEvaluation recall is: %s\n\033[0m" % (test_recall))

        predictions = self.loaded_model.predict(self.x_test)
        y_test = []
        y_pred = []

        if binary:
            y_test = self.y_test
        else:
            y_test = np.argmax(self.y_test, axis=1)

        if binary:
            y_pred = (predictions > 0.5).astype(int).reshape(-1)  # this line is updated
        else:
            y_pred = np.argmax(predictions, axis=1)

        plot_confusion_matrix(
            y_test,
            y_pred,
            class_names=self.class_names,
            path_to_register=self.path_to_register_charts.joinpath(
                "evaluation_metrics/confusion_matrix.png"
            ),
            interactive=self.interactive_reports,
        )

        plot_roc_curve(
            self.y_test,
            predictions,
            class_names=self.class_names,
            binary=self.label_mode == "binary",
            path_to_register=self.path_to_register_charts.joinpath(
                "evaluation_metrics/roc_curve.png"
            ),
            interactive=self.interactive_reports,
        )

        print("\033[94m\nClassification Report:\033[0m")
        print(
            classification_report(
                y_test, y_pred, target_names=self.class_names, zero_division=0
            )
        )

    def predict(self, color="grayscale", img_size=(256, 256), binary=False):
        num_cols = 4
        num_rows = math.ceil(len(self.pred_list) / num_cols)

        plt.figure(figsize=(20, 10))

        for i in range(len(self.pred_list)):
            img = tf.keras.utils.load_img(
                f"data/prediction/{self.pred_list[i]}",
                color_mode=color,
                target_size=img_size,
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = self.loaded_model.predict(img_array)

            score = []

            if binary:
                score = predictions[0][0]
            else:
                score = tf.nn.softmax(predictions[0])

            plt.subplot(num_rows, num_cols, i + 1)
            plt.axis("off")
            plt.imshow(img, cmap="gray")

            if binary:
                if score > 0.5:
                    plt.title("Pneumonia ({:.2f}%)".format(100 * score))
                else:
                    plt.title("Normal ({:.2f}%)".format(100 * (1 - score)))
            else:
                plt.title(
                    "{} ({:.2f}%)".format(
                        self.class_names[np.argmax(score)], 100 * np.max(score)
                    )
                )
        plt.savefig(
            self.path_to_register_charts.joinpath("evaluation_metrics/predictions.png")
        )
        if self.interactive_reports:
            plt.show()
