from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train
from autorecsys.utils.common import set_device
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # set GPU devices
    set_device('gpu:0')

    # load dataset
    used_column = [0, 1, 2]
    record_defaults = [tf.int32, tf.int32, tf.float32]
    data = tf.data.experimental.CsvDataset("./tests/datasets/ml-1m/ratings.dat", record_defaults,
                                           field_delim=",", select_cols=used_column)
    data = data.repeat().shuffle(buffer_size=1000).batch(batch_size=1024)

    # # build recommender
    config_filename = "./examples/configs/mf_config.yaml"
    model = Recommender(config_filename)
    #
    # # train model
    model, avg_loss = train(model, data)
