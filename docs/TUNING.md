# Tune hyperparameters on the default training dataset

&nbsp;

## Introduction

You can tune hyperparameters on the default training dataset before performing cross-validation.

A common approach is to split your dataset into training and validation sets, train your model with different hyperparameter combinations, and choose the combination that performs best on the validation set.

Once you've found the best hyperparameters, you can use cross-validation to evaluate the final model.

Keep in mind that grid search can be computationally expensive, especially if you have a large number of hyperparameters to tune.

You may want to consider using other search methods like random search or Bayesian optimization for more efficient hyperparameter tuning.

&nbsp;

## Code

[See full code here](https://github.com/ArthurB2077/intro-tensorflow-mnist-classification)

```python
import tensorflow as tf
import keras_tuner as kt

"""
Function:
    Helper function that defined the hyperparameters search values intervals. 
    These are the hyperparameters values that will be tested to fin an optimal model.

Args:
    hp (kt.HyperParameters): Hyperparameters that will best fit the model. Used at the kt.Hyperband instanciation.
"""
def model_builder(hp):
    """
    Sequential models are a powerful tool for handling sequential data and can be applied to a wide range of machine learning problems.
    This model is a type of neural network that processes input data in a sequential order, 
    typically one item at a time, and maintains a hidden state that allows the network to learn and remember patterns in the input over time.
    """
    model = tf.keras.Sequential()

    """
    The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images 
    from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). 
    Think of this layer as unstacking rows of pixels in the image and lining them up. 
    This layer has no parameters to learn; it only reformats the data.
    """
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    """
    Tune the number of possible units in the first Dense layer.
    During the Hypertunning search process, optimal number of units and activation function will be selected.
    """
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_activation = hp.Choice("activation", ["relu", "tanh"])

    model.add(tf.keras.layers.Dense(units=hp_units, activation=hp_activation))

    """
    Tune whether to use dropout.
    During the Hypertunning search process, a dropout layer will be add or not in order to optimize the model.
    """
    if hp.Boolean("dropout"):
        model.add(tf.keras.layers.Dropout(rate=0.25))

    """
    The last layer returns a logits array with length of 10.
    Each node contains a score that indicates the current image belongs to one of the 10 classes.
    """
    model.add(tf.keras.layers.Dense(10))

    """
    Tune the learning rate for the optimizer.
    During the Hypertunning search process, optimal learning rate value between 0.0001 and 0.01 will be find.
    """
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling="log")
    """
    This is how the model is updated based on the data it sees and its loss function.
    """
    optimizer_func = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    """
    This measures how accurate the model is during training. 
    You want to minimize this function to "steer" the model in the right direction.
    """
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    """
    Metrics are used to monitor the training and testing steps. 
    The following example uses accuracy, the fraction of the images that are correctly classified.
    """
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=['accuracy'])

    return model

"""
Class:
    Allow to automatically find the optimum hyperparameters given interval values.
    If no search is saved, a new search begin. This step can take some time but when it's done once, it isn't needded to be repeated.
    You can find the previous search result in the hypertunning_logs directory.
    When it's already searched, the model can be instanciate and use for fitting or validation immediatetly.
    This class also dispose of a method that allow to run the model until a maximum defined epochs in order to find the best possible number of epochs to train the model.

Args:
    x_train (numpy.ndarray): Training digit image dimension values.
    y_train (numpy.ndarray): Training digit image label values.
    x_test (numpy.ndarray): Testing digit image dimension values.
    y_test (numpy.ndarray): Testing digit image label values.
    max_epochs (int): maximum numbers of epoch that the model will first initially train to find the best number of epoch for training.
    hypermodel (tf.keras.Model): Model with the best optimized hyperparameters
"""
class HyperModel:
    def __init__(self, x_train, y_train, x_test,y_test, max_epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.max_epochs = max_epochs
        self.hypermodel = None

    """
    Definition:
        Hyperband tuning algorithm: uses adaptive resource allocation and early-stopping to quickly converge on a high-performing model.
            This is done using a sports championship style bracket.
            The algorithm trains a large number of models for a few epochs and carries forward only the top-performing half of models to the next round.
            Hyperband determines the number of models to train in a bracket by computing 1 + logfactor(max_epochs) and rounding it up to the nearest integer.
    Method:
        Initialize the search, if necessary, and build the model with optimum hyperparameters.

    Args:
        self (tf.keras.Model): Model with the best optimized hyperparameters
    """
    def build(self):
        # Create a callback to stop training early after reaching a certain value for the validation loss.
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        hyper_band = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=50, factor=3, directory='hypertunning_logs', project_name='hyperband_algo')
        # Run the hyperparameter search. The arguments for the search method are the same as those used for tf.keras.model.fit in addition to the callback above.
        hyper_band.search(self.x_train, self.y_train, validation_split=0.3, callbacks=[stop_early], epochs=self.max_epochs)
        # Get the best results
        hyper_band.results_summary(1)
        # Get the best hyperparameters.
        best_hyperparameters = hyper_band.get_best_hyperparameters()[0]
        # Build the hypermodel
        hypermodel = hyper_band.hypermodel.build(best_hyperparameters)
        self.hypermodel = hypermodel

        return hypermodel

    """
    Method:
        Find the optimal number of epochs to train the model with the hyperparameters obtained from the search.

    Args:
        self (tf.keras.Model): Model with the best optimized hyperparameters
    """
    def get_best_epoch(self):
        if (self.hypermodel):
            print('\nInitial training... Searching best number of epochs...')
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            initial_training = self.hypermodel.fit(self.x_train, self.y_train, validation_split=0.3, callbacks=[stop_early], epochs=self.max_epochs)
            accurancy_per_epoch = initial_training.history['val_accuracy']
            best_epoch = accurancy_per_epoch.index(
                max(accurancy_per_epoch)) + 1
            print('\nBest epoch is: %d' % (best_epoch))

            return best_epoch
        else:
            return None
```
