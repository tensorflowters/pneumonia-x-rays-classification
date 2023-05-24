# Pnemonia x-rays classification

&nbsp;

## Requirements

- Use a **train-validation-test** procedure

- Use a **cross validation** procedure, compare your results with a simple train test split

- Use one of the datasets to **tune** your algorithms

- Explore and test various **methods** and **compare results** (optimization, feature engineering, metrics, PCA)

> Using a **clear and concise** way to present results should always preval

&nbsp;

## Deliveries

1. Technical documents
    - a Jupyter notebook-like file, containing code and text, possibly graphics
    - a html-file to prove our results without rerunning the code

2. Synthesis document
    - a pdf file to sum up our results and figures

> There are ways to save a trained algorithm and load it afterwards in order to obtain the same results when you run it again !

&nbsp;

## Metrics

After implementing these methods, we'll compare their performances using metrics such as:

- Accuracy

- Precision

- Recall

- F1-score

- Area under the receiver operating characteristic curve (ROC AUC)

This will give us a better understanding of which method works best for our specific problem.

&nbsp;

## Methods

### Cross validation

We can use k-fold cross-validation, where you split your training dataset into k subsets (folds).

Train the model on k-1 folds and validate on the remaining fold, repeating this process k times.

This will help us assess the model's performance more reliably and tune its hyperparameters.

&nbsp;

### Hyper tuning

We can use some hyper tuninng technics to find the best optimum parameters onthe training dataset

&nbsp;

### Convolutional Neural Networks (CNN)

CNNs are the most popular method for image classification tasks.

They learn spatial hierarchies in the data by using convolutional layers, which are especially good at detecting patterns in images.

We can start with a simple architecture and later experiment with deeper and more complex networks.

&nbsp;

### Regularization

To avoid overfitting, consider using regularization techniques such as L1 or L2 regularization, dropout, or batch normalization.

These methods can help our model generalize better to the validation and test datasets.

&nbsp;

### Data Augmentation

 To increase the variability in our dataset and improve the model's generalization, we can apply data augmentation techniques.

 Some common technics include rotations, translations, scaling, flips, and color jittering.

 This can help the model learn more robust features and prevent overfitting.

&nbsp;

### Transfer Learning

We can leverage pre-trained models (such as VGG, ResNet, Inception, etc.) that have already been trained on large image datasets (e.g., ImageNet).

Fine-tune the pre-trained model on our specific task by either updating the last few layers or training the entire network with our smaller dataset.

This can give we a good starting point and potentially better performance.

&nbsp;

## Commands

### Activate Miniconda env

```bash
conda activate zoidberg2.0
```

&nbsp;

### Configuring env variables

#### 1. Listing environment variables

```bash
conda env config vars list
```

#### 2. Setting **TF_CPP_MIN_LOG_LEVEL** in order to deactivate *tensorflow* warning logs

```bash
conda env config vars set TF_CPP_MIN_LOG_LEVEL=2
```

#### 3. Reloading the env

```bash
conda activate zoidberg2.0
```

&nbsp;

### Add a library (pip)

```bash
make addPyLib LIB_NAME=my_library
```
