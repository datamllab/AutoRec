from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train
from autorecsys.utils import load_config

if __name__ == "__main__":
    # set GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs: {}".format(gpus))
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(gpus[7], 'GPU')


    # load dataset
    used_column = [0, 1, 2]
    record_defaults = [tf.int32, tf.int32, tf.float32]
    data = tf.data.experimental.CsvDataset("./tests/datasets/ml-1m/ratings.dat", record_defaults,
                                           field_delim=",", select_cols=used_column)
    data = data.repeat().shuffle(buffer_size=1000).batch(batch_size=1024)


    # for step, (user_id, item_id, y) in enumerate(data.take(100)):
    #     print("test")

    # # build recommender
    config_filename = "mf_config"
    model = Recommender(config_filename)
    #
    # # train model
    # model, avg_loss = train(model, data)


#
# def train(model, data, train_config=None):
#     if train_config is None:
#         train_config = load_config("train_default_config")
#
#     lr = train_config["TrainOption"]["learning_rate"]
#     if isinstance(lr, int):
#         lr = lr
#     elif isinstance(lr, list) and len(lr) == 1:
#         lr = lr[0]
#
#     num_batch = train_config["TrainOption"]["epoch"]
#
#     optimizer = tf.optimizers.Adam(learning_rate=lr)
#
#     avg_loss = []
#
#     for step, (user_id, item_id, y) in enumerate(data.take(num_batch)):
#         print("my test")
#         # feat_dict = {"user_id": user_id, "item_id": item_id}
#         # with tf.GradientTape() as tape:
#         #     y_pred = model(feat_dict)
#         #     loss = tf.keras.losses.MSE(y_pred["rating"], y)
#         # grads = tape.gradient(loss, model.trainable_variables)
#         #
#         # optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         # avg_loss.append(float(loss))
#         # if train_config["TrainOption"]["logging_config"]["freq"] > 0 \
#         #         and step % train_config["TrainOption"]["logging_config"]["freq"] == 0:
#         #     print("Step: {}, avg_loss: {}, loss: {}".format(
#         #         step,
#         #         sum(avg_loss[-1000:]) / min(1000., step + 1),
#         #         loss
#         #         )
#         #     )
#     return model, avg_loss
#
train(model, data)