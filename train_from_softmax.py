#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 0

import math
import datetime
import sys
import argparse
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

from model.balance_input_fn import dataset_pipeline
from model.triplet_loss import batch_hard_triplet_loss
from model.utils import Param


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="config path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg()
    config_path = args.cfg
    params = Param(config_path)

    # dataset
    train_ds, train_count = dataset_pipeline(params)

    # create model
    baseModel = MobileNetV2(include_top=False,
                            weights=None,
                            input_shape=(224, 224, 3),
                            pooling="avg")
    fc = tf.keras.layers.Dense(params.NUM_CLASSES,
                               activation="softmax", 
                               name="dense_final")(baseModel.output)
    model = Model(inputs=baseModel.input, outputs=fc)
    #  model.load_weights("./experiment/01_from_soft_to_tl/epoch-00")

    preLayer = model.get_layer("global_average_pooling2d")
    fc = tf.keras.layers.Dense(128,
                               activation=None,
                               name="embeddings")(preLayer.output)
    l2 = tf.math.l2_normalize(fc)

    model = Model(inputs=baseModel.input, outputs=l2)
    model.load_weights("./experiment/from_soft_to_tl_01/epoch-49")
    model.summary()

    optimizer = tf.keras.optimizers.Adam(params.LR)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss, hpd, hnd = batch_hard_triplet_loss(labels,
                                                     predictions,
                                                     params.MARGIN)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, hpd, hnd

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = config_path + "logs/" + current_time
    file_writer = tf.summary.create_file_writer(logdir+'/train')
    file_writer.set_as_default()

    total_step = 0
    # start training
    for epoch in range(params.EPOCHS):
        for step, (images, labels) in enumerate(train_ds):
            total_step += 1
            loss, hpd, hnd = train_step(images, labels)
            template = "Epoch: {}/{}, step: {}/{}, loss: {:.5f}"
            print(template.format(epoch,
                  params.EPOCHS,
                  step,
                  math.ceil(train_count/
                  (params.NUM_CLASSES_PER_BATCH*params.NUM_IMAGES_PER_CLASS)),
                  loss.numpy()))
            tf.summary.scalar('loss', loss, step=total_step)
            tf.summary.scalar("hardest_positive_dist", hpd, step=total_step)
            tf.summary.scalar("hardest_negative_dist", hnd, step=total_step)

        if epoch % params.save_every_n_epoch == 0:
            model.save_weights(config_path+f"epoch-{epoch}", save_format='tf')
