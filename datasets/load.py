import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


train_dir = pathlib.Path('chest_Xray/train')
test_dir = pathlib.Path('chest_Xray/test')
val_dir = pathlib.Path('chest_Xray/val')

training_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

class_names=training_dataset.class_names

print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in training_dataset.take(3):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[int(labels[i].numpy()[0])])
    print(images[i])
    # print(labels[i].numpy())
    plt.axis("off")
plt.show()

for image_batch, labels_batch in training_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = training_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))