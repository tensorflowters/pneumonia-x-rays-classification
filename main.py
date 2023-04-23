import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import os
import pymongo
from datasets.load import process_directory, create_tf_dataset

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir = pathlib.Path('chest_Xray/train')
test_dir = pathlib.Path('chest_Xray/test')
val_dir = pathlib.Path('chest_Xray/val')

training_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=None,
    image_size=(256, 256),
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=None,
    image_size=(256, 256),
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=None,
    image_size=(256, 256),
    shuffle=True
)



class_names=training_dataset.class_names

training_dataset = training_dataset.batch(32, drop_remainder=True)
test_dataset = test_dataset.batch(32, drop_remainder=True)
validation_dataset = validation_dataset.batch(32, drop_remainder=True)


print("\n\033[91m###SCRIPT LOGS###\033[0m")
print("Training  dataset:\n")
print(training_dataset)
print(training_dataset.options)
# print(class_names)
print(training_dataset.element_spec)
for element in training_dataset.take(1):
    print(element[0].shape) # Inputs
    print(element[1].shape) # Labels
print(len(list(training_dataset.as_numpy_iterator())))
print("You may get surprised to see 163 and not 5216 but remember that we batched our dataset into batch of 32 elements. 5216/32 = 163")
batch_sizes = [batch.shape[0] for _, batch in training_dataset]
plt.bar(range(len(batch_sizes)), batch_sizes)
plt.xlabel('Batch number')
plt.ylabel('Batch size')
plt.show()
print("\n###LOGGER END###")

y_test = []
for _, label in test_dataset.unbatch().as_numpy_iterator():
    y_test.append(label[0])
# print(y_labels)
# print(len(y_labels ))

y_train = []
for _, label in training_dataset.unbatch().as_numpy_iterator():
    y_train.append(label[0])
# print(y_train)
# print(len(y_train ))

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
sns.countplot(x=y_train)
plt.title("Frequency distribution (training dataset)")
plt.xlabel("Binary categorical values")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.countplot(x=y_test)
plt.title("Frequency distribution (testing dataset)")
plt.xlabel("Binary categorical values")
plt.ylabel("Frequency")

plt.show()


plt.figure(figsize=(10, 10))
for images, labels in training_dataset.take(3):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(int(labels[i].numpy()[0]))
    # plt.title(class_names[int(labels[i].numpy()[0])])
    # print(images[i])
    # print(labels[i].numpy())
    plt.axis("off")
plt.show()

# for image_batch, labels_batch in training_dataset:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = training_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))