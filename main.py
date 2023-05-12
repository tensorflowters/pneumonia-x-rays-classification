import pathlib
import numpy as np
import tensorflow as tf

from scripts.basic_model import BasicModel


print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")


model = BasicModel()

# model.get_image()
# model.get_distribution()
# model.get_mean()

model.train(10)

model.evaluate()


print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")