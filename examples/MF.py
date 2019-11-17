# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
import logging
import numpy as np

from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.auto_search import CFRSearch
from autorecsys.pipeline import StructuredDataInput, LatentFactorMapper, MLPInteraction, RatingPredictionOptimizer

from autorecsys.utils.common import set_device
from autorecsys.pipeline.preprocesser import data_load_from_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def custom_pipeline():
    # set GPU devices
    set_device('cpu:0')

    #config filename
    config_filename = "./examples/old/configs/data_default_config.yaml"

    # load dataset
    train_X, train_y, val_X, val_y = data_load_from_config(config_filename)

    # Build the pipeline.
    input_node = StructuredDataInput(column_names=['user_id', 'item_id'])
    # cpu_num should default to None.
    user_emb = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=10)(input_node)
    item_emb = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=10)(input_node)

    mlp_output1 = MLPInteraction(units=hp_module.Choice('units', [32, 64]),
                                num_layers=hp_module.Choice('num_layers', [1, 2]),
                                use_batchnorm=False,
                                dropout_rate=hp_module.Choice('dropout_rate', [0.0, 0.1, 0.5]))([user_emb, item_emb])

    mlp_output2 = MLPInteraction(units=hp_module.Choice('units', [128, 256]),
                                num_layers=hp_module.Choice('num_layers', [3, 4]),
                                use_batchnorm=False,
                                dropout_rate=hp_module.Choice('dropout_rate', [0.0, 0.1, 0.5]))([mlp_output1])

    final_output = RatingPredictionOptimizer()(mlp_output2)

    # AutoML search and predict.
    cf_searcher = CFRSearch(tuner='random',
                            tuner_params={'max_trials': 1, 'overwrite': True},
                            inputs=input_node,
                            outputs=final_output)
    cf_searcher.search(x=train_X, y=train_y, x_val=val_X, y_val=val_y, objective='mse')
    print(cf_searcher.predict(x=val_X, id_column='id', outputs=[mlp_output1, mlp_output2, final_output]))
    print(cf_searcher.evaluate(x=val_X, y_true=val_y))


if __name__ == "__main__":
    custom_pipeline()
