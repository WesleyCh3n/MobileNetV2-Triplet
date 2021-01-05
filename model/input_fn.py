import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #Select GPU device
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error

import tensorflow as tf
import pathlib
import random
from model.utils import Param


_INPUT_SIZE = 0
_BATCH_SIZE = 0
_EPOCHS = 0
_NUM_CLASSES = 0



def parse_filename(path):
    ds_root = pathlib.Path(path)
    filenames = list(ds_root.glob('*/*'))
    filenames = [str(path) for path in filenames]
    random.shuffle(filenames)
    label_names = sorted(item.name for item in ds_root.glob('*/')
            if item.is_dir())
    label_to_index = dict((name, index) for index, name
            in enumerate(label_names))
    labels = [label_to_index[pathlib.Path(path).parent.name]
            for path in filenames]
    return filenames, labels, len(filenames)


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    image = tf.image.resize(image, _INPUT_SIZE)

    # label = tf.one_hot(label, depth=_NUM_CLASSES)
    return image, label


def train_preprocess(image, label):
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def generate_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    # dataset = dataset.repeat(_EPOCHS)
    dataset = dataset.batch(_BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset

def parse_config(params):
    global _INPUT_SIZE, _BATCH_SIZE, _EPOCHS, _NUM_CLASSES
    _INPUT_SIZE = params.INPUT_SIZE
    _BATCH_SIZE = params.BATCH_SIZE
    _EPOCHS = params.EPOCHS
    _NUM_CLASSES =  params.NUM_CLASSES

def dataset_pipeline(params, val=False):
    parse_config(params)
    if val:
        # train_filenames, train_labels, train_counts = parse_filename(params.DATASET+"train/")
        # val_filenames, val_labels, val_counts = parse_filename(params.DATASET+"valid/")
        # train_ds = generate_dataset(train_filenames, train_labels)
        # val_ds = generate_dataset(val_filenames, val_labels)
        # return train_ds, val_ds, train_counts, val_counts
        test_filenames, test_labels, test_counts = parse_filename(params.test_ds)
        test_ds = generate_dataset(test_filenames, test_labels)
        return test_ds, test_counts
    train_filenames, train_labels, train_counts = parse_filename(params.DATASET)
    train_ds = generate_dataset(train_filenames, train_labels)
    return train_ds, train_counts

if __name__ == "__main__":
    config_path = "./experiment/01_cross_incepv4/"
    params = Param(config_path)
    train_ds, train_counts = dataset_pipeline(params)
    print(train_ds, train_counts)

    print("\n\n\n==Finish==\n\n\n")
