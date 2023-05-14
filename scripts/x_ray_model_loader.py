import pathlib
import os
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from x_ray_data_viz import plot_confusion_matrix, plot_roc_curve
from x_ray_dataset_builder import Dataset


class ModelLoader:
    def __init__(self, btch_size=32, img_size=(180, 180)):
        test_dir = pathlib.Path("data/test")
        pred_list = os.listdir("data/prediction/")

        test_ds = Dataset(test_dir, image_size=img_size, batch_size=btch_size)

        AUTOTUNE = tf.data.AUTOTUNE
        test_ds.build(AUTOTUNE)

        self.loaded_model = None
        self.probability_model = None
        self.class_names = test_ds.get_class_names()
        self.pred_list = pred_list
        self.test_ds = test_ds.normalized_dataset
        self.x_test = test_ds.x_dataset
        self.y_test = test_ds.y_dataset
    
    def load(self, model_pathname):
        self.loaded_model = tf.keras.models.load_model(model_pathname)
        self.probability_model = tf.keras.Sequential([self.loaded_model, tf.keras.layers.Softmax()])
    
    def evaluate(self):
        print('\nEvaluating loaded model...')
        print('\n')
        test_loss, categorical_accuracy, test_precision, test_recall, test_auc = self.loaded_model.evaluate(self.test_ds, verbose=2)
        print('\nTest loaded loss is: %s' % (test_loss))
        print('\nTest loaded categorical accurancy is: %s' % (categorical_accuracy))
        print('\nTest loaded precision is: %s' % (test_precision))
        print('\nTest loaded recall is: %s' % (test_recall))
        print('\nTest loaded area under the curve is: %s' % (test_auc))

        y_pred = self.loaded_model.predict(self.x_test)

        true_labels = np.argmax(self.y_test, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)
        plot_confusion_matrix(true_labels, pred_labels, class_names=self.class_names)
        plot_roc_curve(self.y_test, y_pred, class_names=self.class_names)
        print("\nClassification Report:\n")
        print(classification_report(true_labels, pred_labels, target_names=self.class_names, zero_division=0))

    def predict(self, color="grayscale", img_size=(180, 180)):
        num_cols = 4
        num_rows = math.ceil(len(self.pred_list) / num_cols)

        plt.figure(figsize=(20, 10))

        for i in range(len(self.pred_list)):
            img = tf.keras.utils.load_img(
                f"data/prediction/{self.pred_list[i]}",
                color_mode=color,
                target_size=img_size
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = self.loaded_model.predict(img_array)

            score = tf.nn.softmax(predictions[0])
            
            plt.subplot(num_rows, num_cols, i+1)
            plt.axis("off")
            plt.imshow(img, cmap='gray')
            plt.title("{} ({:.2f}%)"
                .format(self.class_names[np.argmax(score)], 100 * np.max(score)))
            
        plt.show()
