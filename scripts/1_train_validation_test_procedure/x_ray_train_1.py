import os
import pathlib

from dotenv import load_dotenv
from x_ray_model_1 import Model

MODEL_ID = os.getenv("MODEL_ID")
BATCH_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_BATCH_SIZE"))
CLASS_TYPE = os.getenv(f"MODEL_{MODEL_ID}_CLASS_TYPE")
EPOCHS = int(os.getenv(f"MODEL_{MODEL_ID}_EPOCHS"))
IMG_COLOR = os.getenv(f"MODEL_{MODEL_ID}_IMG_COLOR")
IMG_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_IMG_SIZE"))
INTERACTIVE_SESSION = bool(int(os.getenv(f"INTERACTIVE_SESSION")))

print(
    "\033[91m"
    "=================================================================\n"
    "****************************TRAINING LOGS************************\n"
    "================================================================="
    "\033[0m\n"
)

model = Model(BATCH_SIZE, IMG_SIZE, IMG_COLOR, CLASS_TYPE, INTERACTIVE_SESSION)

model.train(EPOCHS)

print(
    "\n\033[91m"
    "=================================================================\n"
    "****************************TRAINING LOGS************************\n"
    "================================================================="
    "\033[0m"
)
