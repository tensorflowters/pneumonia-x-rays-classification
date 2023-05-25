import os
import pathlib
import re
import subprocess
import sys

import tensorflowjs as tfjs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)
from dotenv import load_dotenv
from logger.log import text_info, title_important, title_info
from utils.x_ray_model_loader import ModelLoader
from x_ray_model_7 import MergeLayer, Model

load_dotenv(pathlib.Path("./../configuration/.env"))

PYTHON_VERSION = os.getenv("PYTHON_VERSION")
TRAINING_MODE = int(os.getenv("TRAINING_MODE"))
TEST_MODE = int(os.getenv("TEST_MODE"))
EVAL_LOG_FILE = os.getenv(f"EVAL_LOG_FILE")
TRAIN_LOG_FILE = os.getenv(f"TRAIN_LOG_FILE")
BATCH_SIZE = int(os.getenv(f"BATCH_SIZE"))
CLASS_TYPE = os.getenv(f"CLASS_TYPE")
EPOCHS = int(os.getenv(f"EPOCHS"))
IMG_COLOR = os.getenv(f"IMG_COLOR")
IMG_SIZE = int(os.getenv(f"IMG_SIZE"))
INTERACTIVE_SESSION = bool(int(os.getenv(f"INTERACTIVE_SESSION")))

if TRAINING_MODE:
    K_FOLD = int(os.getenv(f"K_FOLD"))

    title_important(message="TRAINING LOGS")

    model = Model(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        img_color=IMG_COLOR,
        label_mode=CLASS_TYPE,
        interactive_reports=INTERACTIVE_SESSION,
    )

    model.train(EPOCHS, K_FOLD)

    title_important(message="TRAINING LOGS")
else:
    METRICS_DIR = pathlib.Path(os.getenv("METRICS_DIR")).absolute()
    DIR = pathlib.Path(os.getenv("DIR")).absolute()
    WEB_DIR = pathlib.Path(os.getenv("WEB_DIR")).absolute()

    title_important("INFERENCE LOGS")

    model_loader = ModelLoader(
        batch_size=BATCH_SIZE,
        color=IMG_COLOR,
        img_size=IMG_SIZE,
        interactive_reports=INTERACTIVE_SESSION,
        label_mode=CLASS_TYPE,
        path_to_register_charts=METRICS_DIR,
    )

    custom_layer = {"ConcatenationLayer": MergeLayer, "MergeLayer": MergeLayer}

    # if(OVERRIDE_BEST):
    #     model_loader.load(OVERRIDE_BEST_FILE_NAME, custom_objects=custom_layer,)
    # else:
    model_loader.load(
        DIR.joinpath(f"model_fold_{BEST_ID}.keras").absolute(),
        custom_objects=custom_layer,
    )

    model_loader.evaluate(batch_size=1, binary=CLASS_TYPE == "binary")

    text_info(message="Predicting...")

    model_loader.predict(
        binary=CLASS_TYPE == "binary",
        color=IMG_COLOR,
        img_size=(IMG_SIZE, IMG_SIZE),
    )

    text_info(message="Predictions done !")

    text_info(message="Exporting to web...")

    tfjs.converters.save_keras_model(model_loader.loaded_model, WEB_DIR)

    text_info(message="Exporting done !")

    title_important("INFERENCE LOGS")
