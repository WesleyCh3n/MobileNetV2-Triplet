#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error

import cv2
import pathlib
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from model.utils import Param, arg


def cvPreprocess(filename):
    image = cv2.imread(filename).astype(np.float32)
    image = ((image/255.0)-0.5)*2.0
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, 0)
    return image

def tfPreprocess(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image-0.5)*2.0
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    args = arg()
    config_path = args.cfg
    params = Param(config_path)

    logdir = os.path.join(config_path, "tflite/")
    filepath = os.path.join(logdir, f"{args.epoch}model.tflite")
    model = tf.lite.Interpreter(model_path=filepath)
    model.allocate_tensors()

    input_details = model.get_input_details()[0]["index"]
    output_details = model.get_output_details()[0]["index"]
    #  print("output_details", output_details)

    #  imagePath="/home/ubuntu/dataset/DATASET/train/Cow00/Cow00_0_51.jpg"
    #  tfImg = tfPreprocess(imagePath)

    #  model.set_tensor(input_details, tfImg)
    #  model.invoke()
    #  output_data1 = model.get_tensor(output_details)
    #  print(output_data1)

    vecs = np.zeros((1900,128))
    metas = np.zeros((1900))
    for label in tqdm(range(19)):
        ds_root = pathlib.Path(f"/home/ubuntu/dataset/test_100/Cow{label:02d}/")
        filenames = list(ds_root.glob('**/*'))
        filenames = [str(path) for path in filenames]
        for i, imagePath in enumerate(filenames):
            cvImg = cvPreprocess(imagePath)
            model.set_tensor(input_details, cvImg)
            model.invoke()
            vecs[label*100+i] = np.round(model.get_tensor(output_details), 1)[0]
            metas[label*100+i] = label


    np.savetxt(os.path.join(logdir, f"tflite-{args.epoch:02d}vecs.tsv"),
               vecs,
               fmt='%.1f',
               delimiter='\t')
    np.savetxt(os.path.join(logdir, f"tflite-{args.epoch:02d}metas.tsv"),
               metas,
               fmt='%i',
               delimiter='\t')
