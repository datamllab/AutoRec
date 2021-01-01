# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import tensorflow as tf
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, DenseFeatureMapper, SparseFeatureMapper, \
                        ElementwiseInteraction, FMInteraction, MLPInteraction, ConcatenateInteraction, \
                        CrossNetInteraction, SelfAttentionInteraction, HyperInteraction, \
                        PointWiseOptimizer
from autorecsys.pipeline.interactor import InnerProductInteraction
from autorecsys.pipeline.preprocessor import CriteoPreprocessor, AvazuPreprocessor
from autorecsys.recommender import CTRRecommender

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_dlrm(emb_dict):
    if 'user' in emb_dict or 'item' in emb_dict:
        emb_list = [emb for _, emb in emb_dict.items()]
        output = MLPInteraction(num_layers=2)(emb_list)
    else:
        sparse_feat_mlp_output = [MLPInteraction()( [emb_dict['sparse']] )] if 'sparse' in emb_dict else []
        dense_feat_mlp_output = [MLPInteraction()( [emb_dict['dense']] )] if 'dense' in emb_dict else []
        output = MLPInteraction(num_layers=2)(sparse_feat_mlp_output + dense_feat_mlp_output)
    return output


def build_deepfm(emb_dict):
    if 'user' in emb_dict or 'item' in emb_dict:
        emb_list = [emb for _, emb in emb_dict.items()]
        fm_output = [FMInteraction()(emb_list)]
        bottom_mlp_output = [MLPInteraction(num_layers=2)(emb_list)]
        output = MLPInteraction(num_layers=2)(fm_output + bottom_mlp_output)
    else:
        fm_output = [FMInteraction()( [emb_dict['sparse']] )] if 'sparse' in emb_dict else []
        bottom_mlp_output = [MLPInteraction()( [emb_dict['dense']] )] if 'dense' in emb_dict else []
        output = MLPInteraction(num_layers=2)(fm_output + bottom_mlp_output)
    return output


def build_crossnet(emb_dict):
    if 'user' in emb_dict or 'item' in emb_dict:
        emb_list = [emb for _, emb in emb_dict.items()]
        fm_output = [CrossNetInteraction()(emb_list)]
        bottom_mlp_output = [MLPInteraction(num_layers=2)(emb_list)]
        output = MLPInteraction(num_layers=2)(fm_output + bottom_mlp_output)
    else:
        fm_output = [CrossNetInteraction()( [emb_dict['sparse']] )] if 'sparse' in emb_dict else []
        bottom_mlp_output = [MLPInteraction()( [emb_dict['dense']] )] if 'dense' in emb_dict else []
        output = MLPInteraction(num_layers=2)(fm_output + bottom_mlp_output)
    return output


def build_autoint(emb_dict):
    if 'user' in emb_dict or 'item' in emb_dict:
        emb_list = [emb for _, emb in emb_dict.items()]
        fm_output = [SelfAttentionInteraction()(emb_list)]
        bottom_mlp_output = [MLPInteraction(num_layers=2)(emb_list)]
        output = MLPInteraction(num_layers=2)(fm_output + bottom_mlp_output)
    else:
        fm_output = [SelfAttentionInteraction()( [emb_dict['sparse']] )] if 'sparse' in emb_dict else []
        bottom_mlp_output = [MLPInteraction()( [emb_dict['dense']] )] if 'dense' in emb_dict else []
        output = MLPInteraction(num_layers=2)(fm_output + bottom_mlp_output)
    return output


def build_neumf(emb_dict):
    emb_list = [emb for _, emb in emb_dict.items()]
    innerproduct_output = [InnerProductInteraction()(emb_list)]
    mlp_output = [MLPInteraction(num_layers=2)(emb_list)]
    output = innerproduct_output + mlp_output
    return output


def build_autorec(emb_dict):
    if 'user' in emb_dict or 'item' in emb_dict:
        emb_list = [emb for _, emb in emb_dict.items()]
        output = HyperInteraction()(emb_list)
    else:
        sparse_feat_bottom_output = [HyperInteraction(meta_interator_num=2)([sparse_feat_emb])] if 'sparse' in emb_dict else []
        dense_feat_bottom_output = [HyperInteraction(meta_interator_num=2)([dense_feat_emb])] if 'dense' in emb_dict else []
        top_mlp_output = HyperInteraction(meta_interator_num=2)(sparse_feat_bottom_output + dense_feat_bottom_output)
        output = HyperInteraction(meta_interator_num=2)([top_mlp_output])
    return output


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='input a model name', default='dlrm')
    parser.add_argument('-data', type=str, help='dataset name', default="avazu")
    parser.add_argument('-data_path', type=str, help='dataset path', default='./example_datasets/avazu/train-10k')
    parser.add_argument('-sep', type=str, help='dataset sep')
    parser.add_argument('-search', type=str, help='input a search method name', default='random')
    parser.add_argument('-batch_size', type=int, help='batch size', default=256)
    parser.add_argument('-trials', type=int, help='try number', default=2)
    parser.add_argument('-gpu_index', type=int, help='the index of gpu to use', default=0)
    args = parser.parse_args()
    print("args:", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    if args.sep == None:
        args.sep = '::'

    if args.data == "avazu":
        # Step 1: Preprocess data
        avazu = AvazuPreprocessor(csv_path=args.data_path, validate_percentage=0.1, test_percentage=0.1)
        train_X, train_y, val_X, val_y, test_X, test_y = avazu.preprocess()
        train_X_categorical = avazu.get_x_categorical(train_X)
        val_X_categorical = avazu.get_x_categorical(val_X)
        test_X_categorical = avazu.get_x_categorical(test_X)
        categorical_count = avazu.get_categorical_count()
        hash_size = avazu.get_hash_size()

        # Step 2: Build the recommender, which provides search space
        # Step 2.1: Setup mappers to handle inputs
        # dense_input_node = None
        sparse_input_node = Input(shape=[categorical_count])
        input = [sparse_input_node]

        # dense_feat_emb = None
        sparse_feat_emb = SparseFeatureMapper(
            num_of_fields=categorical_count,
            hash_size=hash_size,
            embedding_dim=64)(sparse_input_node)

        emb_dict = {'sparse': sparse_feat_emb}

    if args.data == "criteo":
        # Step 1: Preprocess data
        criteo = CriteoPreprocessor(csv_path=args.data_path, validate_percentage=0.1, test_percentage=0.1)
        train_X, train_y, val_X, val_y, test_X, test_y = criteo.preprocess()
        train_X_numerical, train_X_categorical = criteo.get_x_numerical(train_X), criteo.get_x_categorical(train_X)
        val_X_numerical, val_X_categorical = criteo.get_x_numerical(val_X), criteo.get_x_categorical(val_X)
        test_X_numerical, test_X_categorical = criteo.get_x_numerical(test_X), criteo.get_x_categorical(test_X)
        numerical_count = criteo.get_numerical_count()
        categorical_count = criteo.get_categorical_count()
        hash_size = criteo.get_hash_size()

        # Step 2: Build the recommender, which provides search space
        # Step 2.1: Setup mappers to handle inputs
        dense_input_node = Input(shape=[numerical_count])
        sparse_input_node = Input(shape=[categorical_count])
        input = [dense_input_node, sparse_input_node]

        dense_feat_emb = DenseFeatureMapper(
            num_of_fields=numerical_count,
            embedding_dim=64)(dense_input_node)

        sparse_feat_emb = SparseFeatureMapper(
            num_of_fields=categorical_count,
            hash_size=hash_size,
            embedding_dim=64)(sparse_input_node)

        emb_dict = {'dense': dense_feat_emb, 'sparse': sparse_feat_emb}

    # Step 2.2: Setup interactors to handle models
    if args.model == 'dlrm':
        output = build_dlrm(emb_dict)
    if args.model == 'deepfm':
        output = build_deepfm(emb_dict)
    if args.model == 'crossnet':
        output = build_neumf(emb_dict)
    if args.model == 'autoint':
        output = build_autorec(emb_dict)
    if args.model == 'autorec':
        output = build_autorec(emb_dict)

    # Step 2.3: Setup optimizer to handle the target task
    output = PointWiseOptimizer()(output)
    model = CTRRecommender(inputs=input, outputs=output)

    # Step 3: Build the searcher, which provides search algorithm
    searcher = Search(model=model,
                      tuner=args.search,
                      tuner_params={'max_trials': args.trials, 'overwrite': True}
                      )

    # Step 4: Use the searcher to search the recommender
    start_time = time.time()
    searcher.search(x=train_X,
                    y=train_y,
                    x_val=val_X,
                    y_val=val_y,
                    objective='val_BinaryCrossentropy',
                    batch_size=args.batch_size,
                    epochs=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
                    )
    end_time = time.time()
    print("running time:", end_time - start_time)
    print("args", args)
    logger.info('Validation Accuracy (logloss): {}'.format(searcher.evaluate(x=val_X,
                                                                             y_true=val_y)))

    # Step 5: Evaluate the searched model
    logger.info('Test Accuracy (logloss): {}'.format(searcher.evaluate(x=test_X,
                                                                       y_true=test_y)))
