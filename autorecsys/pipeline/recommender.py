from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.utils import load_config
from autorecsys.pipeline.mapper import build_mappers
from autorecsys.pipeline.interactor import build_interactors
from autorecsys.pipeline.optimizer import build_optimizers
from autorecsys.pipeline.graph import Graph

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseRecommender(Graph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def __init__(self, config):
    #     super(BaseRecommender, self).__init__()
    #
    #     self.config = load_config(config)
    #     self._build()

    def _build(self):

        self.mappers = build_mappers(self.config["Mapper"])
        self.interactors = build_interactors(self.config["Interactor"])
        self.optimizers = build_optimizers(self.config["Optimizer"])

    def call(self, feat_dict):

        # mapping
        mapper_output_dict = {}
        for mapper in self.mappers:
            mapper_output_dict[mapper.config["output"]] = mapper({k: feat_dict[k] for k in mapper.config["input"]})

        # interacting
        interactor_output_dict = {}
        for interactor in self.interactors:
            interactor_output_dict[interactor.config["output"]] = interactor(
                {k: mapper_output_dict[k] for k in interactor.config["input"]}
            )

        # predicting
        y_pred = {}
        for optimizer in self.optimizers:
            y_pred[optimizer.config["output"]] = optimizer(
                {k: interactor_output_dict[k] for k in optimizer.config["input"]}
            )

        return y_pred

    def train(self, train_X, train_y, val_X, val_y, train_config="./examples/configs/train_default_config.yaml"):

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
                y_pred = self.call(trian_feat_dict)
                train_loss = tf.keras.losses.MSE(y_pred["rating"], y)
            grads = tape.gradient(train_loss, self.trainable_variables)

            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            if train_config["TrainOption"]["logging_config"]["freq"] > 0 \
                    and step % train_config["TrainOption"]["logging_config"]["freq"] == 0:
                val_user_id = val_X[::, 0]
                val_item_id = val_X[::, 1]
                val_feat_dict = {"user_id": val_user_id, "item_id": val_item_id}
                val_y_pred = self.call(val_feat_dict)
                val_loss = tf.keras.losses.MSE(val_y_pred["rating"], val_y)
                logging.info(
                    "Step: {:<5d}, train_loss: {:>15.10f}, val_loss: {:>15.10f}".format(step, train_loss, val_loss)
                )
        return train_loss, val_loss

    def validate(self, val_X, val_y, metric=tf.keras.metrics.MeanSquaredError):
        pred_y = self.predict(val_X)
        val_loss = metric(pred_y, val_y)
        return val_loss

    def predict(self, val_data):
        pred = self.cal(val_data)
        return pred


class CFRecommender(BaseRecommender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CTRRecommender(BaseRecommender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
