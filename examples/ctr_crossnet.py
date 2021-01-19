# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import logging
import tensorflow as tf
import autokeras as ak
from autorecsys.pipeline import DenseFeatureMapper, SparseFeatureMapper, CrossNetInteraction, MLPInteraction
from autorecsys.pipeline.preprocessor import CriteoPreprocessor


# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Preprocess data
criteo = CriteoPreprocessor()  # the default arguments are setup to preprocess the Criteo example dataset
train_X, train_y, val_X, val_y, test_X, test_y = criteo.preprocess()
train_X_numerical, train_X_categorical = criteo.get_x_numerical(train_X), criteo.get_x_categorical(train_X)
val_X_numerical, val_X_categorical = criteo.get_x_numerical(val_X), criteo.get_x_categorical(val_X)
test_X_numerical, test_X_categorical = criteo.get_x_numerical(test_X), criteo.get_x_categorical(test_X)
numerical_count = criteo.get_numerical_count()
categorical_count = criteo.get_categorical_count()
hash_size = criteo.get_hash_size()

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
dense_input_node = ak.Input(shape=[numerical_count])
sparse_input_node = ak.Input(shape=[categorical_count])
dense_feat_emb = DenseFeatureMapper(
    num_of_fields=numerical_count,
    embedding_dim=2)(dense_input_node)
sparse_feat_emb = SparseFeatureMapper(
    num_of_fields=categorical_count,
    hash_size=hash_size,
    embedding_dim=2)(sparse_input_node)

# Step 2.2: Setup interactors to handle models
crossnet_output = CrossNetInteraction()([dense_feat_emb, sparse_feat_emb])
bottom_mlp_output = MLPInteraction()([dense_feat_emb])
top_mlp_output = MLPInteraction()([crossnet_output, bottom_mlp_output])

# Step 2.3: Setup optimizer to handle the target task
output = ak.ClassificationHead()(top_mlp_output)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=output,
                          max_trials=2,
                          objective='val_loss',
                          tuner='random',
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical, train_X_categorical],
               y=train_y,
               batch_size=32,
               epochs=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
               )

logger.info('Validation Accuracy (logloss): {}'.format(auto_model.evaluate(x=[val_X_numerical, val_X_categorical],
                                                                           y=val_y)))

# Step 5: Evaluate the searched model
logger.info('Test Accuracy (logloss): {}'.format(auto_model.evaluate(x=[test_X_numerical, test_X_categorical],
                                                                     y=test_y)))
