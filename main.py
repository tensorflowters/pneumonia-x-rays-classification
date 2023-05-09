import pathlib
import tensorflow as tf

from scripts.basic_model import BasicModel


print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")


val_dir = pathlib.Path("data/chest_xray/val")


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    batch_size=None,
    image_size=(256, 256),
    shuffle=True,
)


class_names = validation_dataset.class_names


normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)


normalized_validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Initialize lists for training and testing data
x_val, y_val = [], []

# Retrieve training data
for x, y in normalized_validation_dataset:
    x_val.append(x)
    y_val.append(y[0])


model = BasicModel()

model.get_image()
model.get_distribution()
model.get_mean()

model.train(5, x_val, y_val)

# Evaluating the model with the test datasets



# print("\n\033[92mTraining  dataset:\033[0m")
# print(training_dataset)


# print(f"\nDetected {len(class_names)} classes: {class_names}")


# print("\n\033[92mTraining  dataset specs:\033[0m")
# print(training_dataset.element_spec)


# for element in training_dataset.take(1):
#     print(f"\nThe first input of the training dataset has a shape of {element[0].shape}") # Inputs
#     print(f"\nThe first label of the training dataset has a shape of {element[1].shape}") # Labels


# print(f"\nTraining datasets containing {len(list(training_dataset.as_numpy_iterator()))} elements")


# print(
# """\033[93m
# You may get surprised to see 163 and not 5216 but remember that we batched our dataset into batch of 32 elements.

# 5216/32 = 163\033[0m
# """
# )


# batch_sizes = [batch.shape[0] for _, batch in training_dataset]

# plot_batch_number(batch_sizes, "Number", "Size")


# y_test = []

# for _, label in test_dataset.unbatch().as_numpy_iterator():
#     y_test.append(label[0])


# y_train = []

# for _, label in training_dataset.unbatch().as_numpy_iterator():
#     y_train.append(label[0])


# plot_distribution(y_train, y_test)


# plot_images(training_dataset, class_names)

# first_image = image_batch[0]


# print(f"\033[93mNotice the pixel values are now between {np.min(first_image)} and {np.max(first_image)} (0,1)\033[0m")


print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")