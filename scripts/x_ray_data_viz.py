import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


colors = ["#56f6ff", "#e32440"]


def plot_distribution(labels, dataset_name):
    plt.figure(figsize=(20, 10))
    sns.set_style("whitegrid")
    sns.countplot(x=labels, palette=colors)
    plt.title(f"{dataset_name} - Chest x-rays distribution")
    plt.xlabel("Chest x-ray")
    plt.ylabel("Frequency")
    plt.show()


def plot_mean(labels, class_names, dataset_name):
    unique_labels = np.unique(labels)

    mean_train = [np.mean(labels == label) * 100 for label in unique_labels]

    bar_width = 0.25
    index = np.arange(len(unique_labels))

    plt.figure(figsize=(20, 10))
    plt.style.use("seaborn")
    plt.bar(
        index,
        mean_train,
        bar_width,
        label=class_names,
        tick_label=class_names,
        color=colors,
    )
    plt.xlabel("Xray")
    plt.ylabel("Mean occurence (%)")
    plt.title(f"{dataset_name} - Mean occurence of each x-ray")
    plt.xticks(index, class_names)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels_true, labels_pred, class_names):
    matrix = confusion_matrix(labels_true, labels_pred)

    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")
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


def plot_roc_curve(y_true, y_pred_probs, class_names, binary=False):
    y_true_bin = label_binarize(y_true, classes=class_names)
    n_classes = len(class_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        if binary:
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_probs)
            roc_auc = auc(fpr, tpr)
        else:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    if binary:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
    else:
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_pred_probs.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(20, 10))
    sns.set_style("whitegrid")

    for i in range(n_classes):
        if binary:
            plt.plot(fpr, tpr, label=f"Xray (AUC = {roc_auc:.4f})")
        else:
            plt.plot(
                fpr[i], tpr[i], label=f"Xray {class_names[i]} (AUC = {roc_auc[i]:.4f})"
            )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Receiver Operating Characteristic curve")
    plt.legend(loc="lower right")
    plt.show()
