import pathlib
import numpy as np
import tensorflow as tf

from scripts.basic_model import BasicModel


print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")


# Initialize lists for training and testing data

# Retrieve training data




model = BasicModel()

model.get_image()
model.get_distribution()
model.get_mean()

model.train(20)

model.evaluate()

# Evaluating the model with the test datasets

print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")