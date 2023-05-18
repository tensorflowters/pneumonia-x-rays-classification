from custom_layer import ConcatenationLayer
from x_ray_model_loader import ModelLoader


print("\033[91m====================================================================\033[0m")
print("\033[91m****************************INFERENCE LOGS**************************\033[0m")
print("\033[91m====================================================================\033[0m\n")

custom_objects = {"ConcatenationLayer": ConcatenationLayer}

model = ModelLoader(
    btch_size=32, 
    color="rgb", 
    img_size=(256, 256), 
    label_mode="binary"
)
model.load(
    "notebooks/7_transfer_learning/model_7_checkpoint.keras",
    custom_objects=custom_objects,
)
model.evaluate(binary=True)
model.predict(
    binary=True,
    color="rgb",
    img_size=(256, 256),
)

print("\n\033[91m=================================================================\033[0m")
print("\033[91m****************************INFERENCE LOGS*************************\033[0m")
print("\033[91m===================================================================\033[0m")
