import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

from data_vizualisation import plot_prediction, plot_confusion_matrix, plot_roc_curve


class ModelLoader:
    def __init__(self):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist

        # Retrieve training and testing dataset from the loaded MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        """
        Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
        To do so, divide the values by 255. 
        It's important that the training set and the testing set be preprocessed in the same way.
        """
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.reconstructed_model = None
        self.probability_model = None
    
    def load(self, model_pathname):
        # Reconstruct the model identically to the previous train and save model.
        self.reconstructed_model = tf.keras.models.load_model(model_pathname)
        # Creates a probability model from the trained model and makes predictions on a user-specified image.
        self.probability_model = tf.keras.Sequential([self.reconstructed_model, tf.keras.layers.Softmax()])
    
    def evaluate(self):
        # Evaluating the model with the test datasets
        print('\nEvaluating reconstructed model...')
        print('\n')
        test_loss, test_acc = self.reconstructed_model.evaluate(self.x_test, self.y_test, verbose=2)
        print('\nTest reconstructed loss is: %s' % (test_loss))
        print('\nTest reconstructed accurancy is: %s' % (test_acc))
    
    def predict_one_digit(self):
        # Define the index of the digit image to predict
        pred_index = int(input("\nWhich digit the model need to predict (index required): "))

        if 0 <= pred_index < len(self.x_test):
            """
            Finally, use the trained model to make a prediction about a single image.
            However, tf.keras.models are optimized to make predictions on a batch, or collection, of examples at once. 
            Accordingly, even though you're using a single image, you need to add it to a list.
            So we add the image to a batch where it's the only member.
            """
            pred_img = self.x_test[pred_index]
            pred_img = (np.expand_dims(pred_img, 0))
            label = self.y_test[pred_index]
            # Now predict the correct label for this image:
            pred_single = self.probability_model.predict(pred_img)
            plot_prediction(pred_index, pred_single[0], self.y_test, self.x_test)
        else:
            print(f"\nInvalid index. Please enter a number between 0 and {len(self.x_test) - 1}.")
    
    def predict_all_digits(self):
        """
        Make predictions for all digit images (x values) from 
        the probability model created above in order to 
        display a confusion matrix, ROC curve and a classification report
        """
        preds = self.probability_model.predict(self.x_test)
        # Get the predicted labels
        pred_labels = np.argmax(preds, axis=1)
        # Plot the confusion matrix
        plot_confusion_matrix(self.y_test, pred_labels)
        # Plot the ROC curve
        plot_roc_curve(self.y_test, preds)
        # Print the classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, pred_labels))