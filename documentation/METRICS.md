# Metrics

## Introduction

There are several metrics used to measure the performance of machine learning models,
depending on the type of problem (classification, regression, etc.) and the specific objectives of the analysis.

Some common performance metrics for classification problems include:

- **Accuracy**: The proportion of correctly classified instances out of the total instances in the dataset.\
Accuracy is a simple and intuitive metric, but it can be misleading in cases of imbalanced datasets, where the majority class dominates the minority class.\
In such cases, a high accuracy can be achieved by simply classifying all instances as the majority class.

- **Precision**: The proportion of true positive predictions out of all positive predictions made by the model. Precision is a measure of how well the model correctly identifies positive instances, taking into account false positive predictions.

- **Recall (Sensitivity)**: The proportion of true positive predictions out of all actual positive instances in the dataset. Recall is a measure of the model's ability to identify all positive instances, taking into account false negatives.

- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two metrics. F1-score is useful when both precision and recall are important to the problem, and it is more informative than accuracy for imbalanced datasets.

- **Area Under the Receiver Operating Characteristic (ROC-AUC) Curve**: A measure of the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various classification thresholds. A higher ROC-AUC score indicates better classification performance, and it is more robust to class imbalance compared to accuracy.
