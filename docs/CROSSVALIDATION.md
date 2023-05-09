# Cross validation

&nbsp;

## Introduction

Cross-validation is a technique used to evaluate the performance of a machine learning model by dividing the dataset into multiple smaller sets.

The most common type of cross-validation is k-fold cross-validation.

In k-fold cross-validation, the dataset is divided into k equal parts or "folds."

The model is trained k times, each time using k-1 folds for training and the remaining fold for validation.

The average performance across all k iterations is used as the final evaluation metric.

&nbsp;

## Code

Unfortunately, TensorFlow does not have a built-in function for performing k-fold cross-validation.

However, you can use the KFold class from the sklearn library to achieve this.

Here's how you can modify your code to include 5-fold cross-validation:

1. Import the necessary libraries

    ```python
    from sklearn.model_selection import KFold
    ```

2. Define a function that creates and compiles your model

    ```python
    def create_model():
        model = tf.keras.Sequential([
            # Your model architecture here
        ])
        model.compile(
            # Your model compile configuration here
        )
        return model
    ```

3. Load the entire dataset (train and validation) into a single dataset

    ```python
    all_data_dir = pathlib.Path('chest_Xray/all_data')
    all_dataset = tf.keras.utils.image_dataset_from_directory(
        all_data_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='grayscale',
        batch_size=32,
        image_size=(128, 128),
        shuffle=True
    )
    ```

    > Note: You need to move all your train and validation images into a single directory (all_data_dir) for this step.

4. Convert the dataset into NumPy arrays

    ```python
    all_images, all_labels = [], []
    for images, labels in all_dataset:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())

    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    ```

5. Perform 5-fold cross-validation

    ```python
    k = 5
    num_epochs = 10  # Adjust this based on your needs

    kfold = KFold(n_splits=k, shuffle=True, random_state=1)
    fold = 1

    for train_index, val_index in kfold.split(all_images, all_labels):
        print(f"Processing fold {fold}")
        train_images, val_images = all_images[train_index], all_images[val_index]
        train_labels, val_labels = all_labels[train_index], all_labels[val_index]
        
        # Create and compile the model
        model = create_model()
        
        # Train the model
        model.fit(train_images, train_labels, epochs=num_epochs, validation_data=(val_images, val_labels))
        
        # Increment the fold counter
        fold += 1
    ```

&nbsp;

## Conclusion

This code performs 5-fold cross-validation, splitting your dataset into 5 parts, and using one part for validation while training the model on the other 4 parts.

The process is repeated 5 times, each time using a different fold for validation.
