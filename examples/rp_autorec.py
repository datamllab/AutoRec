# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import logging
import tensorflow as tf
import autokeras as ak
from autorecsys.pipeline import SparseFeatureMapper, DenseFeatureMapper, LatentFactorMapper, RatingPredictionOptimizer, HyperInteraction
from autorecsys.pipeline.preprocessor import MovielensPreprocessor, MovielensPreprocessor2

####
# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load dataset
##Netflix Dataset
# dataset_paths = ["./examples/datasets/netflix-prize-data/combined_data_" + str(i) + ".txt" for i in range(1, 5)]
# data = NetflixPrizePreprocessor(dataset_paths)

# Movielens 1M Dataset
data = MovielensPreprocessor("datasets/movielens_rp/ml-1m/ratings.dat")

##Movielens 10M Dataset
# data = MovielensPreprocessor("./examples/datasets/ml-10M100K/ratings.dat")

##Movielens latest Dataset
# data = MovielensPreprocessor("./examples/datasets/ml-latest/ratings.csv", sep=',')

data.preprocessing(val_test_size=0.1, random_state=1314)
train_X, train_y = data.train_X, data.train_y

val_X, val_y = data.val_X, data.val_y
test_X, test_y = data.test_X, data.test_y
user_num, item_num = data.user_num, data.item_num
logger.info('train_X size: {}'.format(train_X.shape))
logger.info('train_y size: {}'.format(train_y.shape))
logger.info('val_X size: {}'.format(val_X.shape))
logger.info('val_y size: {}'.format(val_y.shape))
logger.info('test_X size: {}'.format(test_X.shape))
logger.info('test_y size: {}'.format(test_y.shape))
logger.info('user total number: {}'.format(user_num))
logger.info('item total number: {}'.format(item_num))
####
# build the pipeline.
input = ak.Input(shape=[2])
# sparse_input = SparseFeatureMapper(
#     num_of_fields=2,
#     embedding_dim=64)(input)
# dense_feat_emb = DenseFeatureMapper(
#     num_of_fields=2,
#     embedding_dim=2)(input)

user_emb = LatentFactorMapper(feat_column_id=0,
                              id_num=user_num,
                              embedding_dim=64)(input)
item_emb = LatentFactorMapper(feat_column_id=1,
                              id_num=item_num,
                              embedding_dim=64)(input)

output1 = HyperInteraction()([input])
output2 = HyperInteraction()([output1, user_emb, item_emb])
output3 = HyperInteraction()([output1, output2, user_emb, item_emb])
output4 = HyperInteraction()([output1, output2, output3, user_emb, item_emb])

output = ak.ClassificationHead()(output1)
# model = CTRRecommender(inputs=[dense_input_node, sparse_input_node], outputs=output)

auto_model = ak.AutoModel(inputs=input,
                          outputs=output,
                          overwrite=True,
                          max_trials=2)
auto_model.fit(train_X, train_y, epochs=2)
print(auto_model.evaluate(val_X, val_y))