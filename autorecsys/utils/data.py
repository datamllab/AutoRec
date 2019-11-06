from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def load_dataset(job_config):
    used_column = [0, 1, 2]
    record_defaults = [tf.int32, tf.int32, tf.float32]
    data = tf.data.experimental.CsvDataset(job_config["DataOption"]["filename"], record_defaults,
                                           field_delim=",", select_cols=used_column)

    data = data.repeat() \
        .shuffle(buffer_size=1000) \
        .batch(batch_size=job_config["TrainOption"]["batch_size"]) \
        .prefetch(buffer_size=5)

    return data
