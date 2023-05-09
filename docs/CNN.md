# CNN

&nbsp;

## Introduction

Convolutional Neural Network (CNN) is a type of neural network that is specifically designed for processing grid-like data, such as images.

In the context of deep learning frameworks like TensorFlow and Keras, CNNs are often built as sequential models, where the layers are stacked one after another in a linear manner.

A typical CNN architecture for image classification includes the following types of layers, usually arranged in a sequential order:

* **Convolutional layers (Conv2D)**: These layers apply convolutional filters to the input, which helps the model learn local features in the image, such as edges, textures, and patterns.

* **Activation layers (ReLU, LeakyReLU, etc.)**: These layers introduce non-linearity to the model, enabling it to learn complex patterns.

* **Pooling layers (MaxPooling2D, AveragePooling2D)**: These layers reduce the spatial dimensions of the feature maps, which helps to reduce the computational complexity and control overfitting.

* **Fully connected layers (Dense)**: These layers are usually added at the end of the network to perform the final classification.\
They take the features learned by the convolutional and pooling layers and combine them to produce the final output.

## Code

Here's an example of a simple CNN architecture using the sequential API in Keras:

```python
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model
```

&nbsp;

## Conclusion

In this example, the CNN architecture consists of three sets of Conv2D and MaxPooling2D layers, followed by a Flatten layer to convert the 3D feature maps into a 1D vector, and two Dense layers for classification.

The input images are assumed to be grayscale with a size of 128x128.

You can modify this architecture to better suit your problem by adjusting the number of layers, filters, kernel sizes, and other hyperparameters.
