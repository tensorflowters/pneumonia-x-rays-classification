import os
import pathlib
import re
import subprocess

from dotenv import load_dotenv

load_dotenv()

PYTHON_VERSION = os.getenv("PYTHON_VERSION")
TRAINING_MODE = int(os.getenv("TRAINING_MODE"))
MODEL_ID = int(os.getenv("MODEL_ID"))
MODEL_DIR_LIST = os.getenv("MODEL_DIR_LIST").split(" ")
EVAL_LOG_FILE = os.getenv(f"MODEL_{MODEL_ID}_EVAL_LOG_FILE")
TRAIN_LOG_FILE = os.getenv(f"MODEL_{MODEL_ID}_TRAIN_LOG_FILE")

script = "train" if TRAINING_MODE else "run"
log_file = TRAIN_LOG_FILE if TRAINING_MODE else EVAL_LOG_FILE
best_model_id = 1

env_vars = os.environ.copy()

if MODEL_ID != 1 and not TRAINING_MODE:
    best_model_id = input("Enter best model fold number:")

    env_vars["BEST_MODEL_ID"] = best_model_id


def remove_ansi_escape_sequences(s):
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", s)


with open(log_file, "w") as outputfile:
    process = subprocess.Popen(
        [
            f"python{PYTHON_VERSION}",
            f"scripts/{MODEL_DIR_LIST[int(MODEL_ID) - 1]}/x_ray_{script}_{MODEL_ID}.py",
        ],
        stdout=subprocess.PIPE,
        env=env_vars,
    )

    for line in process.stdout:
        line = line.decode()
        print(line, end="")
        outputfile.write(remove_ansi_escape_sequences(line).replace("", ""))

    process.wait()
