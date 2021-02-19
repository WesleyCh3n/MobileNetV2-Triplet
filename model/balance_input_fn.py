import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #Select GPU device
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error

import numpy as np
import tensorflow as tf
import pathlib
import random
from model.utils import Param


_INPUT_SIZE = 0
_EPOCHS = 0
_NUM_CLASSES = 0
_NUM_IMAGES_PER_CLASS = 0
_NUM_CLASSES_PER_BATCH = 0


def generator():
    while True:
        # Sample the labels that will compose the batch
        labels = np.random.choice(range(_NUM_CLASSES),
                                  _NUM_CLASSES_PER_BATCH,
                                  replace=False)
        for label in labels:
            for _ in range(_NUM_IMAGES_PER_CLASS):
                yield label


def load_image(filename):
    #Parse label
    label = pathlib.Path(filename.numpy().decode('utf-8')).parent.name.replace("Cow", "")
    label = tf.strings.to_number(label, tf.int32)

    image_string = tf.io.read_file(filename)
    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_ACCURATE')
    #This will convert to float values in [0, 1]
    #  image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    image = tf.image.resize(image, _INPUT_SIZE)

    return image, label

def parse_config(params):
    global _INPUT_SIZE, _EPOCHS, _NUM_CLASSES, _NUM_CLASSES_PER_BATCH, _NUM_IMAGES_PER_CLASS
    _INPUT_SIZE = params.INPUT_SIZE
    _EPOCHS = params.EPOCHS
    _NUM_CLASSES = params.NUM_CLASSES
    _NUM_CLASSES_PER_BATCH = params.NUM_CLASSES_PER_BATCH
    _NUM_IMAGES_PER_CLASS = params.NUM_IMAGES_PER_CLASS

# def dataset_pipeline(path):
def dataset_pipeline(params, val=False):
    parse_config(params)
    if val:
        image_dirs = list(pathlib.Path(params.test_ds).iterdir())
    else:
        image_dirs = list(pathlib.Path(params.DATASET).iterdir())
    #load filenames into dataset
    datasets = [tf.data.Dataset.list_files(f"{image_dir}/*.jpg") for image_dir in image_dirs]
    total_num = len(list(pathlib.Path(params.DATASET).glob('*/*')))
    datasets = [dataset.shuffle(10000) for dataset in datasets]
    choice_dataset = tf.data.Dataset.from_generator(generator, tf.int64)
    dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

    #load into image and label
    dataset = dataset.map(lambda x: tf.py_function(load_image, [x], [tf.float32, tf.int32]))
    batch_size = _NUM_CLASSES_PER_BATCH * _NUM_IMAGES_PER_CLASS
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset, total_num

if __name__ == "__main__":
    # train_ds, val_ds = dataset_pipeline(_DATASET)
    # print(train_ds, val_ds)
    # for _, label in dataset.take(5):
    #     print(label)
    _, count = dataset_pipeline("/home/ubuntu/dataset/train_3000/")
    print("\n\n\n==Finish==\n\n\n")
