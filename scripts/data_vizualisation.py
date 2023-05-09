import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


"""
Function:
    Displays an image from the dataset with its corresponding digit label

Args:
    dataset (numpy.ndarray): The dataset containing images.
    labels (numpy.ndarray): The associated labels in the dataset.
"""
def plot_image(dataset, class_names):
    index = int(input("\nEnter the index of the image you want to display: "))

    if 0 <= index < len(dataset):
        plt.figure(figsize=(12, 10))

        plt.imshow(dataset[index], cmap=plt.cm.binary)

        # Set the grid to the digit image dimension in pixels
        plt.gca().set_xticks([x - 0.5 for x in range(1, 28)], minor=True)
        plt.gca().set_yticks([y - 0.5 for y in range(1, 28)], minor=True)
        plt.grid(which="minor", linestyle="-", linewidth=0.5, color="black")

        plt.xticks([])
        plt.yticks([])

        plt.xlabel(f"Digit label: {class_names[index]}")

        plt.show()
    else:
        print(
            f"\nInvalid index. Please enter a number between 0 and {len(dataset) - 1}."
        )


def plot_images(dataset, class_names, number_of_images=25):
    # This line initializes a new matplotlib figure with a specified size of 10 by 10 inches.
    plt.figure(figsize=(20, 10))
    # This line calculates the number of batches needed to display 50 images, given that each batch contains 9 images.
    batch_count = number_of_images // 9 + 1
    # This line iterates over the first batch neededto display 50 xray images the training_dataset. 
    # The dataset is expected to contain image-label pairs. 
    # The take() function from TensorFlow allows you to take a specified number of elements from the dataset.
    for batch_index, (images, labels) in enumerate(dataset.take(batch_count)):
        # The inner loop now iterates over all the images in the current batch (len(images)), 
        # and the loop stops when it reaches the 50th image.
        for i in range(len(images)):
            # The subplots are now arranged in a 5x10 grid, and the index is updated accordingly.
            ax = plt.subplot(5, 10, i + 1 + (9 * batch_index))
            # This line displays the i-th image in the current batch of images. 
            # It first converts the image data to a NumPy array with the numpy() method,
            # then casts it to the uint8 data type, which is commonly used to represent image pixel values.
            plt.imshow(images[i].numpy().astype("uint8"))
            # This line replaces the previous title with a more human-readable version of the label by using the class_names list.
            # The integer label is used as an index to access the corresponding class name in the class_names list.
            plt.title(class_names[int(labels[i].numpy()[0])])
            # Turns off the axis lines and labels for the current subplot, making the image display cleaner.
            plt.axis("off")
            # The stopping condition is checked so it stops if the 50th image is reached
            if i + 1 + (9 * batch_index) == number_of_images:
                break
    # This line displays the final figure with all the subplots. 
    # It renders the entire grid of images with their corresponding class names as titles.
    plt.show()


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
def plot_confusion_matrix(labels_true, labels_pred):
    matrix = confusion_matrix(labels_true, labels_pred)

    plt.figure(figsize=(16, 10))

    sns.heatmap(
        matrix,
        annot=True,
        cmap="YlGnBu",
        fmt="d",
        xticklabels=range(10),
        yticklabels=range(10),
    )

    plt.title("Confusion matrix")

    plt.xlabel("Predicted digit")
    plt.ylabel("Actual digit")

    plt.show()


"""
Definition:
    The Receiver Operating Characteristic (ROC) curve is a graphical representation that plots the true positive rate (sensitivity) against the false positive rate (1-specificity) at various threshold levels.
    The true positive rate (TPR) is the proportion of actual positive instances that are correctly identified by the classifier, while the false positive rate (FPR) is the proportion of actual negative instances that are incorrectly identified as positive.
    It helps visualize the trade-off between sensitivity (TPR) and specificity (1-FPR) for a classifier.
    A perfect classifier would have an ROC curve that hugs the top-left corner of the plot, indicating a high true positive rate and a low false positive rate.
    A classifier with no predictive power would lie on the diagonal line, representing a random guess.

    The AUC stands for "Area Under the ROC Curve." It is a single scalar value that measures the classifier's overall performance across all threshold levels.
    The AUC ranges from 0 to 1, with a higher value indicating better classifier performance.
    An AUC of 0.5 represents a classifier with no discriminative power, equivalent to random guessing, while an AUC of 1 represents a perfect classifier that makes no mistakes.
    AUC is useful for comparing different classifiers, as it takes into account both the true positive rate and false positive rate.
    It is also less sensitive to class imbalance, which makes it a popular choice for evaluating classification models in real-world applications where class distribution might be skewed.

Function:
    Display a chart representing the predicted result ROC curve and AUC values for each class category.

Args:
    y_true (numpy.ndarray): The true labels for the dataset.
    y_pred_probs (numpy.ndarray): The predicted probabilities for each class.
"""
def plot_roc_curve(y_true, y_pred_probs):
    y_true_bin = label_binarize(y_true, classes=range(10))
    n_classes = y_true_bin.shape[1]

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

    # Compute the AUC value for each digit category
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Digit {i} (AUC = {roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--")

    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.title("Receiver Operating Characteristic curve")

    plt.legend(loc="lower right")

    plt.show()


"""
Function:
    Displays an digit image on which our model has made a prediction.
    Similar to plot_image except that the prediction and validation results are display to

Args:
    i (int): Index of the digit image in the dataset.
    predictions_array (numpy.ndarray): Array of predictions for the digit image.
    true_label (int): Validation test label of the image.
    img (numpy.ndarray): Digit image to be displayed.
"""
def plot_prediction(i, predictions_array, true_label, img):
    plt.figure(figsize=(12, 10))

    true_label, img = true_label[i], img[i]

    # Set the grid to the digit image dimension in pixels
    plt.gca().set_xticks([x - 0.5 for x in range(1, 28)], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, 28)], minor=True)
    plt.grid(which="minor", linestyle="-", linewidth=0.5, color="black")

    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.2f}% ({})".format(
            predicted_label, 100 * np.max(predictions_array), true_label
        ),
        color=color,
    )

    plt.show()
