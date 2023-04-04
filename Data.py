import os

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
img_size = 224
data_dir = "dataset/intel_images"


def load_data(train_dir='train', test_dir='test'):
    print('Loading the data...')

    train_dir_ = os.path.join(data_dir, train_dir)
    test_dir_ = os.path.join(data_dir, test_dir)

    # ------------------------------------
    # Generates batches of augmented images

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.2,  # 20% of the data will be used for validation
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True
    )

    # ------------------------------------
    # Load the data

    train_data = train_datagen.flow_from_directory(
        train_dir_,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_data = train_datagen.flow_from_directory(
        train_dir_,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_data = test_datagen.flow_from_directory(
        test_dir_,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # ---------------------
    # Get the classes
    classes = []
    for class_ in train_data.class_indices:
        classes.append(class_)

    return train_data, val_data, test_data, classes


def show_samples(data, classes):
    print('Plotting samples of data...')
    images, labels = next(data)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        label_index = np.argmax(labels[i])
        ax.set_title(classes[label_index])
        ax.axis('off')

    plt.show()


def get_x_and_y(data):
    print('Extracting images and labels...')
    """
    Return images and labels from a keras DirectoryIterator.
    """
    images = []
    labels = []

    for i in range(len(data)):
        batch_images, batch_labels = next(data)
        images.append(batch_images)
        labels.append(batch_labels)

    # important: transform to flat array
    # so we don't get list of batches
    images = np.concatenate(images)
    labels = np.concatenate(labels)

    return images, labels
