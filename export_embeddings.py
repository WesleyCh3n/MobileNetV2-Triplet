#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 0

import io
import argparse
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:  # Restrict TensorFlow to only use the first GPU
    tf.config.experimental.set_visible_devices(gpus[gpuNum], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpuNum], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from model.utils import Param

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="config path", type=str)
    parser.add_argument("epoch", help="epoch", type=int)
    return parser.parse_args()

def parse_function(filename):
    image_string = tf.io.read_file(filename)
    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    image = tf.image.resize(image, (224, 224))
    return image

if __name__ == "__main__":
    args = arg()
    # load testing dataset
    config_path = args.cfg
    params = Param(config_path)

    # load model
    baseModel = MobileNetV2(include_top=False,
                            weights=None,
                            input_shape=(224, 224, 3),
                            pooling="avg")
    fc = tf.keras.layers.Dense(128,
                               activation=None,
                               name="embeddings")(baseModel.output)
    l2 = tf.math.l2_normalize(fc)

    model = Model(inputs=baseModel.input, outputs=l2)
    model.load_weights(config_path+f"epoch-{args.epoch}")

    # result
    empty_data = np.zeros((100, 19))
    df = pd.DataFrame(data = empty_data, columns = [i for i in range(19)]).astype('object')
    for label in tqdm(range(19)):
        ds_root = pathlib.Path(f"/home/ubuntu/dataset/test_100/Cow{label:02d}/")
        filenames = list(ds_root.glob('**/*'))
        filenames = [str(path) for path in filenames]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(parse_function)
        dataset = dataset.batch(100)
        results = model.predict(dataset)
        for i, emb in enumerate(results):
            df.iloc[i, label] = emb

    logdir = os.path.join(config_path, "emb/")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    pkl_path = os.path.join(logdir, f"{args.epoch}embs.pkl")
    df.to_pickle(pkl_path)

    vecs = np.zeros((1900,128))
    metas = np.zeros((1900))

    for label in tqdm(range(19)):
        for i in range(100):
            vecs[label*100+i] = df[label][i]
            metas[label*100+i] = label

    np.savetxt(os.path.join(logdir, f"{args.epoch:02d}vecs.tsv"),
               vecs,
               fmt='%.10f',
               delimiter='\t')
    np.savetxt(os.path.join(logdir, f"{args.epoch:02d}metas.tsv"),
               metas,
               fmt='%i',
               delimiter='\t')
