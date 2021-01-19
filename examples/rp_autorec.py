# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import logging
import tensorflow as tf
import autokeras as ak
from autorecsys.pipeline import SparseFeatureMapper, HyperInteraction
from autorecsys.pipeline.preprocessor import MovielensPreprocessor

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Preprocess data
movielens = MovielensPreprocessor()
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()
train_X_categorical = movielens.get_x_categorical(train_X)
val_X_categorical = movielens.get_x_categorical(val_X)
test_X_categorical = movielens.get_x_categorical(test_X)
user_num, item_num = movielens.get_hash_size()

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
input = ak.Input(shape=[2])
sparse_input_node = SparseFeatureMapper(
    num_of_fields=2,
    hash_size=[user_num, item_num],
    embedding_dim=32)(input)

# Step 2.2: Setup interactors to handle models
output1 = HyperInteraction()([sparse_input_node])
output2 = HyperInteraction()([output1, sparse_input_node])
output3 = HyperInteraction()([output1, output2, sparse_input_node])
output4 = HyperInteraction()([output1, output2, output3, sparse_input_node])

# Step 2.3: Setup optimizer to handle the target task
output = ak.RegressionHead()(output4)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=input,
                          outputs=output,
                          objective='val_mean_squared_error',
                          max_trials=2,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_categorical],
               y=train_y,
               batch_size=32,
               epochs=2)

logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X_categorical],
                                                                       y=val_y)))

# Step 5: Evaluate the searched model
logger.info('Test Accuracy (mse): {}'.format(auto_model.evaluate(x=[test_X_categorical],
                                                                 y=test_y)))
