# Tune hyperparameters on the default training dataset

&nbsp;

## Introduction

You can tune hyperparameters on the default training dataset before performing cross-validation.

A common approach is to split your dataset into training and validation sets, train your model with different hyperparameter combinations, and choose the combination that performs best on the validation set.

Once you've found the best hyperparameters, you can use cross-validation to evaluate the final model.

Keep in mind that grid search can be computationally expensive, especially if you have a large number of hyperparameters to tune.

You may want to consider using other search methods like random search or Bayesian optimization for more efficient hyperparameter tuning.
