import pathlib
import os
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report
from data_vizualisation import plot_confusion_matrix, plot_roc_curve
from dataset import Dataset


class ModelLoader:
    def __init__(self):
        # Set path for directories including images that will be use to build evaluation and predictions datasets
        test_dir = pathlib.Path("data/test")
        pred_list = os.listdir("data/prediction/")

        test_ds = Dataset(test_dir)

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
        # Reconstruct the model identically to the previous train and save model.
        self.loaded_model = tf.keras.models.load_model(model_pathname)
        # Creates a probability model from the trained model and makes predictions on a user-specified image.
        self.probability_model = tf.keras.Sequential([self.loaded_model, tf.keras.layers.Softmax()])
    
    def evaluate(self):
        """
        Evaluate and display usefull metrics to evaluate model's performances on unseen data (test dataset)
        """
        # Evaluating the model with the test datasets
        print('\nEvaluating loaded model...')
        print('\n')
        test_loss, test_acc = self.loaded_model.evaluate(self.test_ds, verbose=2)
        print('\nTest loaded loss is: %s' % (test_loss))
        print('\nTest loaded accurancy is: %s' % (test_acc))

        y_pred = self.loaded_model.predict(self.x_test)

        # Also convert the one-hot encoded true labels to class labels
        true_labels = np.argmax(self.y_test, axis=1)
        # Get the predicted labels
        pred_labels = np.argmax(y_pred, axis=1)
        # Plot the confusion matrix
        plot_confusion_matrix(true_labels, pred_labels, class_names=self.class_names)
        # Plot the ROC curve
        plot_roc_curve(self.y_test, y_pred, class_names=self.class_names)
        # Print the classification report
        print(
        """
            \nAccuracy: 
            The proportion of correctly classified instances out of the total instances in the dataset.
            Accuracy is a simple and intuitive metric, but it can be misleading in cases of imbalanced datasets, 
            where the majority class dominates the minority class.
            In such cases, a high accuracy can be achieved by simply classifying all instances as the majority class.
        """
        )
        print(
        """
            \nPrecision: 
            The proportion of true positive predictions out of all positive predictions made. 
            Precision is a measure of how well the model correctly identifies positive instances, 
            taking into account false positive predictions.
        """
        )
        print(
        """
            \nRecall (Sensitivity): 
            The proportion of true positive predictions out of all actual positive instances in the dataset. 
            Recall is a measure of the model's ability to identify all positive instances, 
            taking into account false negatives.
        """
        )
        print(
        """
            \nF1-Score: 
            The harmonic mean of precision and recall, providing a balance between the two metrics. 
            F1-score is useful when both precision and recall are important to the problem, and 
            it is more informative than accuracy for imbalanced datasets.
        """
        )
        print("\nClassification Report:\n")
        print(classification_report(true_labels, pred_labels, target_names=self.class_names, zero_division=0))

    def predict(self):
        # Define the number of columns (images per row)
        num_cols = 4
        # Calculate the number of rows needed
        num_rows = math.ceil(len(self.pred_list) / num_cols)

        plt.figure(figsize=(20, 10))

        for i in range(len(self.pred_list)):
            img = tf.keras.utils.load_img(
                f"data/prediction/{self.pred_list[i]}",
                color_mode="grayscale",
                target_size=(512, 512)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = self.loaded_model.predict(img_array)

            score = tf.nn.softmax(predictions[0])
            
            plt.subplot(num_rows, num_cols, i+1)
            plt.axis("off")
            plt.imshow(img)
            plt.title("{} ({:.2f}%)"
                .format(self.class_names[np.argmax(score)], 100 * np.max(score)))
            
        plt.show()
