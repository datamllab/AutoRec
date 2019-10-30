# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.utils import load_config
from autorecsys.pipeline.preprocesser import data_load
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(model, train_X, train_y, val_X, val_y, train_config="./examples/configs/train_default_config.yaml"):


    train_config = load_config(train_config)

    lr = train_config["TrainOption"]["learning_rate"]
    if isinstance(lr, int):
        lr = lr
    elif isinstance(lr, list) and len(lr) == 1:
        lr = lr[0]
    num_batch = train_config["TrainOption"]["epoch"]
    batch_size = train_config["TrainOption"]["batch_size"]
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    data = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    data = data.repeat().shuffle(buffer_size=1000).batch(batch_size=batch_size)

    for step, (X, y) in enumerate(data.take(num_batch)):
        user_id = X[::, 0]
        item_id = X[::, 1]
        trian_feat_dict = {"user_id": user_id, "item_id": item_id}
        with tf.GradientTape() as tape:
            y_pred = model(trian_feat_dict)
            train_loss = tf.keras.losses.MSE(y_pred["rating"], y)
        grads = tape.gradient(train_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if train_config["TrainOption"]["logging_config"]["freq"] > 0 \
                and step % train_config["TrainOption"]["logging_config"]["freq"] == 0:
            val_user_id = val_X[::,0]
            val_item_id = val_X[::,1]
            val_feat_dict = {"user_id": val_user_id, "item_id": val_item_id}
            val_y_pred = model(val_feat_dict)
            val_loss = tf.keras.losses.MSE(val_y_pred["rating"], val_y)
            logging.info(
                "Step: {:<5d}, train_loss: {:>15.10f}, val_loss: {:>15.10f}".format(step, train_loss, val_loss)
                )
    return model, train_loss, val_loss


def validate(model, val_X, val_y, metric=tf.keras.metrics.MeanSquaredError):
    pred_y = predict(model, val_X)
    val_loss = metric(pred_y, val_y)
    return val_loss


def predict(model, val_data):
    pred = model(val_data)
    return pred


# def run(model, train_X, train_y, val_X, val_y, train_config="./examples/configs/train_default_config.yaml"):
#     model, train_loss = train(model, train_X, train_y, train_config)
#     val_loss = validate(model, val_X, val_y)
#     print(val_loss)
#     return model, train_loss, val_loss


if __name__ == "__main__":
    # test()
    pass
