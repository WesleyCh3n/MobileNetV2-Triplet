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
    NUM = 5
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

    empty_data = np.zeros((100, NUM))
    df = pd.DataFrame(data = empty_data, columns = [i for i in range(NUM)]).astype('object')
    for label in tqdm(range(NUM)):
        ds_root = pathlib.Path(f"/home/ubuntu/dataset/test_code/Cow{label:02d}/")
        filenames = list(ds_root.glob('**/*'))
        filenames = [str(path) for path in filenames]
        for i in range(100):
            df.iloc[i, label] = model.predict(cvPreprocess(filenames[i]))

    logdir = os.path.join(config_path, "emb/")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    #  pkl_path = os.path.join(logdir, f"cv-{args.epoch}embs.pkl")
    #  df.to_pickle(pkl_path)

    vecs = np.zeros((NUM*100,128))
    metas = np.zeros((NUM*100))

    for label in tqdm(range(NUM)):
        for i in range(100):
            vecs[label*100+i] = df[label][i]
            metas[label*100+i] = label

    np.savetxt(os.path.join(logdir, f"test-{args.epoch:02d}vecs.tsv"),
               vecs,
               fmt='%.8f',
               delimiter='\t')
    np.savetxt(os.path.join(logdir, f"test-{args.epoch:02d}metas.tsv"),
               metas,
               fmt='%i',
               delimiter='\t')
