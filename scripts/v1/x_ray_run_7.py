import os
import pathlib
import sys

import tensorflowjs as tfjs
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.x_ray_model_loader import ModelLoader
from x_ray_model_7 import MergeLayer

MODEL_ID = os.getenv("MODEL_ID")
BATCH_SIZE = int(os.getenv(f"MODEL_{MODEL_ID}_BATCH_SIZE"))
BEST_MODEL_ID = int(os.getenv("BEST_MODEL_ID"))
CHART_DIR = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_CHART_DIR")).absolute()
CLASS_TYPE = os.getenv(f"MODEL_{MODEL_ID}_CLASS_TYPE")
EPOCHS = int(os.getenv(f"MODEL_{MODEL_ID}_EPOCHS"))
OVERRIDE_BEST = int(os.getenv(f"MODEL_{MODEL_ID}_OVERRIDE_BEST"))
OVERRIDE_BEST_FILE_NAME = pathlib.Path(os.getenv(f"MODEL_{MODEL_ID}_OVERRIDE_BEST_FILE_NAME")).absolute()
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

custom_layer = {"ConcatenationLayer": MergeLayer, "MergeLayer": MergeLayer}

if(OVERRIDE_BEST):
    model_loader.load(OVERRIDE_BEST_FILE_NAME, custom_objects=custom_layer,)
else:
    model_loader.load(MODEL_DIR.joinpath(f"model_7_fold_{BEST_MODEL_ID}.keras").absolute(), custom_objects=custom_layer,)

model_loader.evaluate(batch_size=1, binary=CLASS_TYPE == "binary")

print("\n\033[94mPredicting...\n\033[0m")

model_loader.predict(
    binary=CLASS_TYPE == "binary",
    color=IMG_COLOR,
    img_size=(IMG_SIZE, IMG_SIZE),
)

print("\n\033[94mModel has made predictions !\033[0m")

print("\n\033[94mExporting to web...\n\033[0m")

tfjs.converters.save_keras_model(model_loader.loaded_model, WEB_DIR)

print("\n\033[94mExport done !\033[0m")

print(
    "\n\033[91m"
    "=================================================================\n"
    "***************************INFERENCE LOGS************************\n"
    "================================================================="
    "\033[0m"
)
