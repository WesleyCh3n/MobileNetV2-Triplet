import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 1

import argparse
import pathlib
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

from model.balance_input_fn import dataset_pipeline
from model.triplet_loss import batch_hard_triplet_loss
from model.utils import Param
from tqdm import tqdm

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
    config_path = args.cfg
    params = Param(config_path)

    # create model
    baseModel = MobileNetV2(include_top=False,
                            weights='imagenet',
                            input_shape=(224, 224, 3),
                            pooling="avg")
    fc = tf.keras.layers.Dense(params.NUM_CLASSES, activation="softmax", name="dense_final")(baseModel.output)
    model = Model(inputs=baseModel.input, outputs=fc)
    model.load_weights(os.path.join(config_path, f"epoch-{args.epoch}"))
    #  model.summary()

    #  for label in range(19):
    label = 3
    ds_root = pathlib.Path(f"/home/ubuntu/dataset/train_3000/Cow{label:02d}/")
    filenames = list(ds_root.glob('**/*'))
    filenames = [str(path) for path in filenames]
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(100)
    for item in dataset:
        results = model(item, training=True)
        for result in results:
            print(np.argmax(result))
    #  print(results)
    #  results = model.predict(dataset, traini)
    #  error = 0
    #  for i, result in enumerate(results):
    #      print(result)
    #      print(np.argmax(result))
    #      if np.argmax(result) != label:
    #          error += 1
    #  print(f"{label} acc: {(1-error/100)*100}")
