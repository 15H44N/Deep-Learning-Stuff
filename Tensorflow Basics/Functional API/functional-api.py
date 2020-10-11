import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #no annoyiing messages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
import pandas as pd

# HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001


#data loading part

# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_df = pd.read_csv("archive/train.csv")
test_df = pd.read_csv("archive/test.csv")
train_images = os.getcwd() + "archive/train_images/" + train_df.iloc[:, 0].values
test_images = os.getcwd() + "archive/test_images/" + test_df.iloc[:, 0].values

train_labels = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 1:].values

# We are using 2 output hence we need Functional API
def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)

    # In older versions you need to set shape in order to avoid error
    # on newer (2.3.0+) the following 3 lines can safely be removed
    image.set_shape((64, 64, 1))
    label[0].set_shape([])
    label[1].set_shape([])

    labels = {"first_num": label[0], "second_num": label[1]}
    return image, labels

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = (
    train_dataset.shuffle(buffer_size=len(train_labels))
    .map(read_image)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = (
    test_dataset.map(read_image)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)