import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_vizualisation.display import distribution, mean


class Dataset:
    def __init__(
        self,
        dir_path: str,
        batch_size=40,
        color_mode="grayscale",
        image_size=256,
        label_mode="binary",
        subset=None,
        validation_split=None,
    ):
        self.batch_size = batch_size
        self.class_names = None
        self.color_mode = color_mode
        self.dataset = None
        self.dir_path = dir_path
        self.image_size = image_size
        self.label_mode = label_mode
        self.normalized_dataset = None
        self.raw_dataset = None
        self.raw_x_dataset = []
        self.subset = subset
        self.validation_split = validation_split
        self.x_dataset = []
        self.y_dataset = []

    def build(self, autotune, is_training=False):
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.dir_path,
            batch_size=self.batch_size,
            color_mode=self.color_mode,
            image_size=(self.image_size, self.image_size),
            label_mode=self.label_mode,
            labels="inferred",
            seed=123,
            subset=self.subset,
            validation_split=self.validation_split,
        )

        if is_training:
            self.dataset = dataset.cache().shuffle(1024).prefetch(buffer_size=autotune)
        else:
            self.dataset = dataset.cache().prefetch(buffer_size=autotune)

        self.class_names = dataset.class_names
        self.raw_dataset = dataset

        for x, y in dataset.unbatch().as_numpy_iterator():
            self.raw_x_dataset.append(x)

        self.raw_x_dataset = np.array(self.raw_x_dataset)

        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        self.normalized_dataset = self.dataset.map(
            lambda x, y: (normalization_layer(x), y)
        )

        for x, y in self.normalized_dataset.unbatch().as_numpy_iterator():
            self.x_dataset.append(x)
            self.y_dataset.append(y)

        self.x_dataset, self.y_dataset = np.array(self.x_dataset), np.array(
            self.y_dataset
        )

        return self.normalized_dataset

    def get_class_names(self):
        return self.class_names

    def get_x_batch_shape(self):
        for image_batch, _ in self.dataset:
            return image_batch.shape

    def get_y_batch_shape(self):
        for _, labels_batch in self.dataset:
            return labels_batch.shape

    def display_images_in_batch(
        self,
        batch_index: int,
        dataset_name: str,
        path_to_register: str,
        interactive=True,
    ):
        images, labels = next(iter(self.dataset.take(batch_index)))

        plt.figure(figsize=(20, 10))

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
            plt.title(
                f"{dataset_name} - {self.class_names[np.argmax(labels[i])]} (batch {batch_index})"
            )
            plt.axis("off")

        plt.savefig(path_to_register)

        if interactive:
            plt.show()

    def display_batch_number(
        self, dataset_name: str, path_to_register: str, interactive=True
    ):
        total_images = len(self.x_dataset)
        
        batch_size = self.batch_size

        total_batches = total_images // batch_size

        if total_images % batch_size != 0:
            total_batches += 1

        batch_indices = list(range(total_batches))
        batch_sizes = [batch_size] * total_batches

        if total_images % batch_size != 0:
            batch_sizes[-1] = total_images % batch_size

        plt.figure(figsize=(20, 10))

        plt.style.use("seaborn")

        plt.bar(batch_indices, batch_sizes, color="#ff6f00")

        plt.xlabel("Batchs")

        plt.ylabel("Images")

        plt.title(f"Batchs and images per batch in {dataset_name}")

        plt.savefig(path_to_register)

        if interactive:
            plt.show()

    def display_distribution(
        self, dataset_name: str, path_to_register: str, interactive=True
    ):
        labels_index = np.argmax(self.y_dataset, axis=1)
        labels = []

        for index in labels_index:
            labels.append(self.class_names[index])

        display.distribution(
            labels, dataset_name, path_to_register, interactive=interactive
        )

    def display_mean(self, dataset_name, path_to_register: str, interactive=True):
        labels = np.argmax(self.y_dataset, axis=1)
        display.mean(
            labels,
            self.class_names,
            dataset_name,
            path_to_register,
            interactive=interactive,
        )
