#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error

import cv2
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tqdm import tqdm

from model.utils import Param, arg

def cvPreprocess(filename):
    image = cv2.imread(filename).astype(np.float32)
    image = ((image/255.0)-0.5)*2.0
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, 0)
    return image

if __name__ == "__main__":
    args = arg()
    config_path = args.cfg
    params = Param(config_path)
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

    imagePath="/home/ubuntu/dataset/DATASET/train/Cow00/Cow00_0_51.jpg"

    result = model.predict(cvPreprocess(imagePath))
    print(result)
