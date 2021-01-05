import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 1

import math
import datetime
import sys
import argparse
import numpy as np
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

#  from model.balance_input_fn import dataset_pipeline
from model.input_fn import dataset_pipeline
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
    test_ds, test_count = dataset_pipeline(params, True)

    # create model
    baseModel = MobileNetV2(include_top=False,
                            weights='imagenet',
                            input_shape=(224, 224, 3),
                            pooling="avg")
    fc = tf.keras.layers.Dense(params.NUM_CLASSES,
                               activation="softmax",
                               name="dense_final")(baseModel.output)
    model = Model(inputs=baseModel.input, outputs=fc)
    model.summary()

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    logdir = config_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir+'/train')
    file_writer.set_as_default()

    #  @tf.function(experimental_relax_shapes=True)
    def train_step(images, labels, step):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        #  for pre in predictions:
        #      print(np.argmax(pre))
        #  print("labels: ", labels)
        train_accuracy.update_state(y_true=labels, y_pred=predictions)
        return loss

    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_accuracy.update_state(labels, predictions)

    total_step = 0
    # start training
    for epoch in tf.range(params.EPOCHS):
        for step, (images, labels) in enumerate(train_ds):
            total_step += 1
            loss = train_step(images, labels, total_step)
            template = "Epoch: {}/{}, step: {}/{}, loss: {:.5f}, acc: {:.5f}"
            print(template.format(epoch,
                  params.EPOCHS,
                  step,
                  math.ceil(train_count/
                  (params.NUM_CLASSES_PER_BATCH*params.NUM_IMAGES_PER_CLASS)),
                  loss.numpy(),
                  float(train_accuracy.result())))
            tf.summary.scalar('loss', loss, step=total_step)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=total_step)

        train_accuracy.reset_states()
        for images, labels in test_ds:
            test_step(images, labels)
        print(f"test accuracy: {test_accuracy.result()}")

        test_accuracy.reset_states()

        if epoch % params.save_every_n_epoch == 0:
            model.save_weights(filepath=config_path+"epoch-{}".format(epoch), save_format='tf')
