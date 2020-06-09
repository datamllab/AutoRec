# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import time
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import logging
# logging setting
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


import tensorflow as tf
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, RatingPredictionOptimizer, HyperInteraction, MLPInteraction, \
    ElementwiseInteraction
from autorecsys.pipeline.preprocessor import MovielensPreprocessor, NetflixPrizePreprocessor
from autorecsys.recommender import RPRecommender




def build_mf(user_num, item_num):
    input = Input(shape=[2])
    user_emb = LatentFactorMapper(feat_column_id=0,
                                  id_num=user_num,
                                  embedding_dim=64)(input)
    item_emb = LatentFactorMapper(feat_column_id=1,
                                  id_num=item_num,
                                  embedding_dim=64)(input)
    output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb, item_emb])
    output = RatingPredictionOptimizer()(output)
    model = RPRecommender(inputs=input, outputs=output)
    return model


def build_gmf(user_num, item_num):
    input = Input(shape=[2])
    user_emb = LatentFactorMapper(feat_column_id=0,
                                  id_num=user_num,
                                  embedding_dim=64)(input)
    item_emb = LatentFactorMapper(feat_column_id=1,
                                  id_num=item_num,
                                  embedding_dim=64)(input)
    output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb, item_emb])
    output = RatingPredictionOptimizer()(output)
    model = RPRecommender(inputs=input, outputs=output)
    return model


def build_mlp(user_num, item_num):
    input = Input(shape=[2])
    user_emb_mlp = LatentFactorMapper(feat_column_id=0,
                                      id_num=user_num,
                                      embedding_dim=64)(input)
    item_emb_mlp = LatentFactorMapper(feat_column_id=1,
                                      id_num=user_num,
                                      embedding_dim=64)(input)
    output = MLPInteraction()([user_emb_mlp, item_emb_mlp])
    output = RatingPredictionOptimizer()(output)
    model = RPRecommender(inputs=input, outputs=output)
    return model


def build_neumf(user_num, item_num):
    input = Input(shape=[2])
    user_emb_gmf = LatentFactorMapper(feat_column_id=0,
                                      id_num=user_num,
                                      embedding_dim=64)(input)
    item_emb_gmf = LatentFactorMapper(feat_column_id=1,
                                      id_num=item_num,
                                      embedding_dim=64)(input)
    innerproduct_output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb_gmf, item_emb_gmf])

    user_emb_mlp = LatentFactorMapper(feat_column_id=0,
                                      id_num=user_num,
                                      embedding_dim=64)(input)
    item_emb_mlp = LatentFactorMapper(feat_column_id=1,
                                      id_num=item_num,
                                      embedding_dim=64)(input)
    mlp_output = MLPInteraction()([user_emb_mlp, item_emb_mlp])

    output = RatingPredictionOptimizer()([innerproduct_output, mlp_output])
    model = RPRecommender(inputs=input, outputs=output)
    return model


def build_autorec(user_num, item_num):
    input = Input(shape=[2])
    user_emb_1 = LatentFactorMapper(feat_column_id=0,
                                    id_num=user_num,
                                    embedding_dim=64)(input)
    item_emb_1 = LatentFactorMapper(feat_column_id=1,
                                    id_num=item_num,
                                    embedding_dim=64)(input)

    user_emb_2 = LatentFactorMapper(feat_column_id=0,
                                    id_num=user_num,
                                    embedding_dim=64)(input)
    item_emb_2 = LatentFactorMapper(feat_column_id=1,
                                    id_num=item_num,
                                    embedding_dim=64)(input)

    output = HyperInteraction()([user_emb_1, item_emb_1, user_emb_2, item_emb_2])
    output = RatingPredictionOptimizer()(output)
    model = RPRecommender(inputs=input, outputs=output)
    return model


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='input a model name')
    parser.add_argument('-data', type=str, help='dataset name')
    parser.add_argument('-data_path', type=str, help='dataset path')
    parser.add_argument('-sep', type=str, help='dataset sep')
    parser.add_argument('-search', type=str, help='input a search method name')
    parser.add_argument('-batch_size', type=int, help='batch size')
    parser.add_argument('-epochs', type=int, help='epochs')
    parser.add_argument('-early_stop', type=int, help='early stop')
    parser.add_argument('-trials', type=int, help='try number')
    args = parser.parse_args()
    # print("args:", args)
    if args.sep == None:
        args.sep = '::'

    # Load dataset
    if args.data == "ml":
        data = MovielensPreprocessor(args.data_path, sep=args.sep)
    if args.data == "netflix":
        dataset_paths = [args.data_path + "/combined_data_" + str(i) + ".txt" for i in range(1, 5)]
        data = NetflixPrizePreprocessor(dataset_paths)
    data.preprocessing(val_test_size=0.1, random_state=1314)
    train_X, train_y = data.train_X, data.train_y
    val_X, val_y = data.val_X, data.val_y
    test_X, test_y = data.test_X, data.test_y
    user_num, item_num = data.user_num, data.item_num
    logging.info('train_X size: {}'.format(train_X.shape))
    logging.info('train_y size: {}'.format(train_y.shape))
    logging.info('val_X size: {}'.format(val_X.shape))
    logging.info('val_y size: {}'.format(val_y.shape))
    logging.info('test_X size: {}'.format(test_X.shape))
    logging.info('test_y size: {}'.format(test_y.shape))
    logging.info('user total number: {}'.format(user_num))
    logging.info('item total number: {}'.format(item_num))

    # select model
    if args.model == 'mf':
        model = build_mf(user_num, item_num)
    if args.model == 'mlp':
        model = build_mlp(user_num, item_num)
    if args.model == 'gmf':
        model = build_gmf(user_num, item_num)
    if args.model == 'neumf':
        model = build_neumf(user_num, item_num)
    if args.model == 'autorec':
        model = build_autorec(user_num, item_num)

    # search and predict.
    searcher = Search(model=model,
                      tuner=args.search,  ## hyperband, bayesian
                      tuner_params={'max_trials': args.trials, 'overwrite': True}
                      )

    start_time = time.time()
    searcher.search(x=train_X,
                    y=train_y,
                    x_val=val_X,
                    y_val=val_y,
                    objective='val_mse',
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stop)])
    end_time = time.time()
    # print("Runing time:", end_time - start_time)
    # print("Args", args)
    logging.info('Runing time: {}'.format(end_time - start_time))
    logging.info('Args: {}'.format(args))
    logging.info('Predicting Val Dataset Accuracy (mse): {}'.format(searcher.evaluate(x=val_X, y_true=val_y)))
    logging.info('Predicting Test Dataset Accuracy (mse): {}'.format(searcher.evaluate(x=test_X, y_true=test_y)))
