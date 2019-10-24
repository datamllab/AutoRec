from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # set GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs: {}".format(gpus))
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(gpus[7], 'GPU')

    # load dataset
    used_column = [0, 1, 2]
    record_defaults = [tf.int32, tf.int32, tf.float32]
    data = tf.data.experimental.CsvDataset("./examples/datasets/ml-1m/ratings.dat", record_defaults,
                                           field_delim=",", select_cols=used_column)
    data = data.repeat().shuffle(buffer_size=1000).batch(batch_size=1024)

    # # build recommender
    config_filename = "mf_config"
    model = Recommender(config_filename)
    #
    # # train model
    model, avg_loss = train(model, data)
