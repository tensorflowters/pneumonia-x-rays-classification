# Transfer learning

&nbsp;

## Introduction

Transfer learning involves using a pre-trained model, typically trained on a large dataset like ImageNet, as a starting point for your task.

You can fine-tune the pre-trained model on your dataset, which can save training time and potentially result in better performance.

Popular pre-trained models for image classification include VGG, ResNet, Inception, and MobileNet, among others.

It is recommended to start with transfer learning, as it usually provides a good balance between performance and ease of implementation.

You can experiment with different pre-trained models and fine-tuning strategies.

## Code

To implement transfer learning in TensorFlow, you can use the applications module, which provides pre-trained models with easy-to-use APIs.

Here's an example using MobileNetV2:

```python
from tensorflow.keras.applications import MobileNetV2

def create_model():
    base_model = MobileNetV2(input_shape=(128, 128, 1), include_top=False, weights=None)  # Grayscale images have 1 channel

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model
```

&nbsp;

## Conclusion

This code snippet creates a model based on MobileNetV2, adapted for grayscale images.

The include_top=False parameter removes the final classification layer, and we add our own dense layer with a sigmoid activation function for binary classification.

Since the pre-trained weights are for RGB images, we set weights=None and train the model from scratch on the chest X-ray dataset.
