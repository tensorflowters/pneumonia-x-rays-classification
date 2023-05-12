import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_batch_number(size, x_label, y_label):
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(size)), size)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


"""
Definition:
    Distribution refers to the frequency or proportion of different classes or categories within the dataset.

Function:
    Display 2 charts representing the distribution of each digit category in the training and testing dataset.

Args:
    y_train (numpy.ndarray): Xray labels for the training set.
    y_test (numpy.ndarray): Xray labels for the testing set.
"""
def plot_distribution(y_train, y_test):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title("Xray distribution (training dataset)")
    plt.xlabel("Xray")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test)
    plt.title("Xray distribution (testing dataset)")
    plt.xlabel("Xray")
    plt.ylabel("Frequency")

    plt.show()


"""
Definition:
    The mean refer to a statistical measure of central tendency of the predicted probabilities or scores assigned to each class by the model.
    However, it is important to note that the mean predicted probabilities may not always be the best measure of model performance,
    as it depends on the problem and the specific use case.

Function:
    Display a chart reprsenting the mean occurrence of each label in the training and testing sets.

Args:
    y_train (numpy.ndarray): Xray labels for the training set.
    y_test (numpy.ndarray): Xray labels for the testing set.
"""
def plot_mean(y_train, y_test):
    unique_labels = np.unique(y_train)

    mean_train = [np.mean(y_train == label) * 100 for label in unique_labels]
    mean_test = [np.mean(y_test == label) * 100 for label in unique_labels]

    bar_width = 0.35
    index = np.arange(len(unique_labels))

    plt.figure(figsize=(20, 10))

    plt.bar(index, mean_train, bar_width, label="Training dataset")
    plt.bar(index + bar_width, mean_test, bar_width, label="Testing dataset")

    plt.xlabel("Xray")
    plt.ylabel("Mean occurence (%)")

    plt.title("Mean occurence of each digit in their respective dataset")
    plt.xticks(index + bar_width / 2, unique_labels)
    plt.legend()
    plt.tight_layout()

    plt.show()


"""
Definition:
    A confusion matrix, also known as an error matrix, is a visualization tool used to evaluate the performance of a classification model.
    It is a table that summarizes the number of correct and incorrect predictions made by the model, broken down by each class.
    The confusion matrix helps identify patterns or trends in misclassifications, making it easier to understand the model's strengths and weaknesses.

Function:
    Display a chart representing the predicted result confusion maxtrix.

Args:
    labels_true (numpy.ndarray): The true labels for the dataset.
    labels_pred (numpy.ndarray): The predicted labels for the dataset.
"""
def plot_confusion_matrix(labels_true, labels_pred, class_names):
    matrix = confusion_matrix(labels_true, labels_pred)

    plt.figure(figsize=(16, 10))

    sns.heatmap(
        matrix,
        annot=True,
        cmap="YlGnBu",
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion matrix")

    plt.xlabel("Predicted results")
    plt.ylabel("Actual results")

    plt.show()


"""
Definition:
    The Receiver Operating Characteristic (ROC) curve is a graphical representation that plots the true positive rate (sensitivity) 
    against the false positive rate (1-specificity) at various threshold levels.
    
    The true positive rate (TPR) is the proportion of actual positive instances that are correctly identified by the classifier, 
    while the false positive rate (FPR) is the proportion of actual negative instances that are incorrectly identified as positive.
    
    It helps visualize the trade-off between sensitivity (TPR) and specificity (1-FPR) for a classifier.
    
    ROC AUC score can be high while your other metrics are not as good. 
    It's important to consider all these metrics together when evaluating your model, 
    and you may need to adjust your decision threshold depending on which metrics are most important for your specific application.
    
    The ROC AUC score, however, takes into account all possible thresholds, not just the one that was used to create the confusion matrix and 
    the classification report. 
    So, even though the model is not performing well at the current threshold, it may be performing better at other thresholds.
Function:
    Display a chart representing the predicted result ROC curve and AUC values for each class category.

Args:
    y_true (numpy.ndarray): The true labels for the dataset (one-hot encoded).
    y_pred_probs (numpy.ndarray): The predicted probabilities for each class.
"""
def plot_roc_curve(y_true, y_pred_probs, class_names):
    y_true_bin = label_binarize(y_true, classes=class_names)
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot the ROC curve
    plt.figure(figsize=(20, 10))

    # Compute the AUC value for each category
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Xray {class_names[i]} (AUC = {roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--")

    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.title("Receiver Operating Characteristic curve")

    plt.legend(loc="lower right")

    plt.show()