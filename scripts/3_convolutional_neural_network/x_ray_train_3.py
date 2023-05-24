import os
import pathlib

from dotenv import load_dotenv
from x_ray_model_3 import Model

MODEL_ID = os.getenv("MODEL_ID")
BATCH_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_BATCH_SIZE"))
CLASS_TYPE = os.getenv(f"MODEL_{MODEL_ID}_CLASS_TYPE")
EPOCHS = int(os.getenv(f"MODEL_{MODEL_ID}_EPOCHS"))
IMG_COLOR = os.getenv(f"MODEL_{MODEL_ID}_IMG_COLOR")
IMG_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_IMG_SIZE"))
K_FOLD = int(os.getenv(f"MODEL_{MODEL_ID}_K_FOLD"))
INTERACTIVE_SESSION = bool(int(os.getenv(f"INTERACTIVE_SESSION")))
MAX_EPOCHS = int(os.getenv(f"MODEL_{MODEL_ID}_MAX_EPOCHS"))

print(
    "\033[91m"
    "=================================================================\n"
    "****************************TRAINING LOGS************************\n"
    "================================================================="
    "\033[0m\n"
)

model = Model(
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    img_color=IMG_COLOR,
    label_mode=CLASS_TYPE,
    interactive_reports=INTERACTIVE_SESSION,
)

model.train(EPOCHS, MAX_EPOCHS, K_FOLD)

print(
    "\033[91m"
    "=================================================================\n"
    "****************************TRAINING LOGS************************\n"
    "================================================================="
    "\033[0m\n"
)