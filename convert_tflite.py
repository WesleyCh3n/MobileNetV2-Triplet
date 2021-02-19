#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 0

import argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
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

    model._set_inputs(inputs=tf.random.normal(shape=(1, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3)))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    logdir = os.path.join(config_path, "tflite/")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    filepath = os.path.join(logdir, f"{args.epoch}model.tflite")
    open(filepath, "wb").write(tflite_model)
    print("Done")
