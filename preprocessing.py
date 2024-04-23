import os
import pickle
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input
import glob

def data_load(dataset_dir, batch_size, img_size):
    train = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training',
    )

    test = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation',
    )

    train = train.map(
        lambda x, y: (preprocess_input(x), y))
    test = test.map(
        lambda x, y: (preprocess_input(x), y))

    return train, test