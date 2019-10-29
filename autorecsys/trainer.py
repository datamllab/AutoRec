from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.utils import load_config

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def train(model, data, train_config="./examples/configs/train_default_config.yaml"):

    train_config = load_config(train_config)

    lr = train_config["TrainOption"]["learning_rate"]
    if isinstance(lr, int):
        lr = lr
    elif isinstance(lr, list) and len(lr) == 1:
        lr = lr[0]

    num_batch = train_config["TrainOption"]["epoch"]

    optimizer = tf.optimizers.Adam(learning_rate=lr)

    avg_loss = []

    for step, (user_id, item_id, y) in enumerate(data.take(num_batch)):
        feat_dict = {"user_id": user_id, "item_id": item_id}
        with tf.GradientTape() as tape:
            y_pred = model(feat_dict)
            loss = tf.keras.losses.MSE(y_pred["rating"], y)
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        avg_loss.append(float(loss))
        if train_config["TrainOption"]["logging_config"]["freq"] > 0 \
                and step % train_config["TrainOption"]["logging_config"]["freq"] == 0:
            logging.info("Step: {}, avg_loss: {}, loss: {}".format(
                step,
                sum(avg_loss[-1000:]) / min(1000., step + 1),
                loss
            )
            )
    return model, avg_loss
