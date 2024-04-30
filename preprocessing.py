import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def data_load(dataset_dir, batch_size, img_size):
    train_dir = os.path.join(dataset_dir, "train")
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
    )

    val_dir = os.path.join(dataset_dir, "val")
    val = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
    )

    test_dir = os.path.join(dataset_dir, "test")
    test = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
    )

    train = train.map(lambda x, y: (preprocess_input(x), y))
    val = val.map(lambda x, y: (preprocess_input(x), y))
    test = test.map(lambda x, y: (preprocess_input(x), y))

    return train, val, test
