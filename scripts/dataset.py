import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_vizualisation import plot_distribution, plot_mean

class Dataset:
    def __init__(self, dir_path, validation_split=None, subset=None, color_mode="grayscale", batch_size=64, image_size=(512, 512)):
        self.dir_path = dir_path
        self.validation_split = validation_split
        self.subset = subset
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset = None
        self.normalized_dataset = None
        self.class_names = None
        self.x_dataset = []
        self.y_dataset = []

    def build(self, autotune, is_training=False):
        """
        Function:
            Load images from disk and build an image dataset that instanciate tf.data.Dataset class from Tensorflow API.
            
        Arguments:
            test_dir (pathlib.Path):
                Directory where the data is located. 
                If labels is "inferred", it should contain subdirectories, each containing images for a class. 
                Otherwise, the directory structure is ignored.
            labels (str):
                Either "inferred" (labels are generated from the directory structure), None (no labels), 
                or a list/tuple of integer labels of the same size as the number of image files found in the directory. 
                Labels should be sorted according to the alphanumeric order of the image file paths (obtained via os.walk(directory) in Python).
            label_mode (str):
                String describing the encoding of labels.
                Options are:
                    - 'int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
                    - 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss).
                    - 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy).
                    - None (no labels).
                A chest ray is either pneumonia or normal. 'binary' mode will be appropriate.
            color_mode (str):
                One of "grayscale", "rgb", "rgba". Default: "rgb". 
                Whether the images will be converted to have 1, 3, or 4 channels.
                In the case chest ray, grayscale will be enough because haave no impact in pneumonia detection.
            seed (int):
                Optional random seed for shuffling and transformations.
            batch_size (int):
                Size of the batches of data. Default: 32. 
                If None, the data will not be batched (the dataset will yield individual samples).

            image_size (int):
                Size to resize images to after they are read from disk, specified as (height, width). 
                Defaults to (256, 256). 
                Since the pipeline processes batches of images that must all have the same size, this must be provided.
        """
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.dir_path,
            labels="inferred",
            label_mode="categorical", # the labels are a float32 tensor of shape (batch_size, num_classes), representing a one-hot encoding of the class index.
            validation_split=self.validation_split,
            subset=self.subset,
            seed=123,
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            image_size=self.image_size,
        )

        # Argument autotune need to be tf.data.AUTOTUNE but just defined once during script execution
        if(is_training):
            self.dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
        else:
            self.dataset = dataset.cache().prefetch(buffer_size=autotune)

        self.class_names = dataset.class_names

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.normalized_dataset = self.dataset.map(lambda x, y: (normalization_layer(x), y))
        
        for x, y in self.normalized_dataset.unbatch().as_numpy_iterator():
            self.x_dataset.append(x)
            self.y_dataset.append(y)
        
        self.x_dataset, self.y_dataset = np.array(self.x_dataset), np.array(self.y_dataset)

        return self.normalized_dataset
    
    def get_class_names(self):
        return self.class_names
    
    def get_x_batch_shape(self):
        for image_batch, _ in self.dataset:
            return image_batch.shape
    
    def get_y_batch_shape(self):
        for _, labels_batch in self.dataset:
            return labels_batch.shape
        
    def display_images_in_batch(self, batch_index, dataset_name):
        images, labels = next(iter(self.dataset.take(batch_index)))

        plt.figure(figsize=(20, 10))

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(f"{dataset_name} - {self.class_names[np.argmax(labels[i])]} (batch {batch_index})")
            plt.axis("off")
        
        plt.show()

    def display_batch_number(self, dataset_name):
        # Count the total number of images
        total_images = len(self.x_dataset)
        batch_size = self.batch_size  # Or whatever batch size you're using

        # Calculate the number of batches
        total_batches = total_images // batch_size

        # There might be one more batch if the total number of images isn't divisible by the batch size
        if total_images % batch_size != 0:
            total_batches += 1

        batch_indices = list(range(total_batches))
        batch_sizes = [batch_size]*total_batches

        # If the total number of images isn't divisible by the batch size, 
        # the last batch will contain the remainder
        if total_images % batch_size != 0:
            batch_sizes[-1] = total_images % batch_size


        plt.figure(figsize=(20, 10))
        plt.style.use('seaborn')
        plt.bar(batch_indices, batch_sizes, color='#ff6f00')
        plt.xlabel('Batchs')
        plt.ylabel('Images')
        plt.title(f'Batchs and images per batch in {dataset_name}')
        plt.show()

    def display_distribution(self, dataset_name):
        # Display x-ray distribution in dataset
        labels_index = np.argmax(self.y_dataset, axis=1)
        labels = []

        for index in labels_index:
            labels.append(self.class_names[index])
        
        plot_distribution(labels, dataset_name)

    def display_mean(self, dataset_name):
        # Display x-ray mean occurence in dataset
        labels = np.argmax(self.y_dataset, axis=1)
        plot_mean(labels, self.class_names, dataset_name)
