## code=utf-8

import os
import yaml
import logging
import tensorflow as tf
from autorecsys.config import mapper_config, interaction_config, recommender_config

logger = logging.getLogger(__name__)


class MF(tf.keras.Model):
    def __init__(self, config_filename):
        super(MF, self).__init__()
        with open(os.join("./config/", config_filename), "r", encoding='utf-8') as fr:
            config = yaml.load(fr)
            print(config)
            print(config["Mapper"])
            print(config["Interaction"])
            print(config["Optimizer"])

        print(mapper_config(config["Mapper"]))

        self.mapper_dict = mapper_config(config["Mapper"])
        self.interaction_dict = interaction_config(config["Interaction"], self.mapper_dict)
        self.optimizer_dict = {}

        self.user_latentfactorMapper = self.mapper_dict["user_id"]
        self.item_latentfactorMapper = self.mapper_dict["item_id"]

        # self.interaction = InnerProductInteraction()
        self.interaction = MLPInteraction()
        for interaction_name in self.interaction_dict:
            self.interaction_dict[interaction_name]["InteractionType"] = self.interaction_dict[interaction_name][
                "InteractionType"]()
        print(self.interaction_dict)

        self.optimizer = RatingPredictionOptimizer()

    def call(self, userID, ItemID):
        mapper_out_dict = {"user_id": self.mapper_dict["user_id"](userID),
                           "item_id": self.mapper_dict["item_id"](ItemID)}

        interaction_out_dict = {}

        for interaction_name in self.interaction_dict:
            interaction_out_dict[interaction_name] = self.interaction(
                [mapper_out_dict[x] for x in interaction_dict[interaction_name]["Input"]])
        y_pred = [interaction_out_dict[x] for x in interaction_out_dict]
        y_pred = self.optimizer(y_pred)
        return y_pred


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    used_column = [0, 1, 2]
    record_defaults = [tf.int32, tf.int32, tf.float32]

    data = tf.data.experimental.CsvDataset("./tests/datasets/ml-1m/ratings.dat", record_defaults, field_delim=",",
                                           select_cols=used_column)
    data = data.repeat().shuffle(buffer_size=1000).batch(batch_size=10240).prefetch(buffer_size=5)

    # Cross-Entropy Loss.
    model = MF("config.yaml")
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    avg_loss = []

    for step, (user_id, item_id, y) in enumerate(data.take(100000)):
        with tf.GradientTape() as tape:
            y_pred = model(user_id, item_id)
            loss = tf.keras.losses.MSE(y_pred, y)
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        avg_loss.append(float(loss))
        print(step, "avg_loss", sum(avg_loss[-1000:]) / min(1000., step + 1), 'loss:', float(loss))
