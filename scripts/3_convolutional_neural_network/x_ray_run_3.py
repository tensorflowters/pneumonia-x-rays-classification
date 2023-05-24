import os
import pathlib
import sys

import tensorflowjs as tfjs
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.x_ray_model_loader import ModelLoader

MODEL_ID = os.getenv("MODEL_ID")
BATCH_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_BATCH_SIZE"))
BEST_MODEL_ID = int(os.getenv("BEST_MODEL_ID"))
CHART_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_CHART_DIR")).absolute()
CLASS_TYPE = os.getenv(f"MODEL_{MODEL_ID}_CLASS_TYPE")
EPOCHS = int(os.getenv(f"MODEL_{MODEL_ID}_EPOCHS"))
IMG_COLOR = os.getenv(f"MODEL_{MODEL_ID}_IMG_COLOR")
IMG_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_IMG_SIZE"))
INTERACTIVE_SESSION = bool(int(os.getenv(f"INTERACTIVE_SESSION")))
MODEL_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_MODEL_DIR")).absolute()
WEB_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_WEB_DIR")).absolute()


print(
    "\n\033[91m"
    "=================================================================\n"
    "***************************INFERENCE LOGS************************\n"
    "================================================================="
    "\033[0m\n"
)

model_loader = ModelLoader(
    batch_size=BATCH_SIZE,
    color=IMG_COLOR,
    img_size=IMG_SIZE,
    interactive_reports=INTERACTIVE_SESSION,
    label_mode=CLASS_TYPE,
    path_to_register_charts=CHART_DIR,
)

model_loader.load(MODEL_DIR.joinpath(f"model_3_fold_{BEST_MODEL_ID}.keras").absolute())

model_loader.evaluate(batch_size=BATCH_SIZE, binary=CLASS_TYPE == "binary")

model_loader.predict(
    binary=CLASS_TYPE == "binary",
    color=IMG_COLOR,
    img_size=(IMG_SIZE, IMG_SIZE),
)

tfjs.converters.save_keras_model(model_loader.loaded_model, WEB_DIR)

print(
    "\n\033[91m"
    "=================================================================\n"
    "***************************INFERENCE LOGS************************\n"
    "================================================================="
    "\033[0m\n"
)