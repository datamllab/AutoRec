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
    innerproduct_output = [ElementwiseInteraction(elementwise_type="innerporduct")(emb_list)]
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
    parser.add_argument('-data_path', type=str, help='dataset path', default='./datasets/avazu/avazu_500K.txt')
    parser.add_argument('-sep', type=str, help='dataset sep')
    parser.add_argument('-search', type=str, help='input a search method name', default='random')
    parser.add_argument('-batch_size', type=int, help='batch size', default=256)
    parser.add_argument('-trials', type=int, help='try number', default=10)
    parser.add_argument('-gpu_index', type=int, help='the index of gpu to use', default=0)
    args = parser.parse_args()
    print("args:", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    if args.sep == None:
        args.sep = '::'

    # Load and preprocess dataset
    if args.data == "avazu":

        avazu = AvazuPreprocessor(csv_path=args.data_path, validate_percentage=0.1, test_percentage=0.1)
        train_X, train_y, val_X, val_y, test_X, test_y = avazu.preprocess()

        # dense_input_node = None
        sparse_input_node = Input(shape=[avazu.get_categorical_count()])
        input = [sparse_input_node]

        # dense_feat_emb = None
        sparse_feat_emb = SparseFeatureMapper(
            num_of_fields=avazu.get_categorical_count(),
            hash_size=avazu.get_hash_size(),
            embedding_dim=64)(sparse_input_node)

        emb_dict = {'sparse': sparse_feat_emb}


    if args.data == "criteo":

        criteo = CriteoPreprocessor(csv_path=args.data_path, validate_percentage=0.1, test_percentage=0.1)
        train_X, train_y, val_X, val_y, test_X, test_y = criteo.preprocess()

        # build the pipeline.
        dense_input_node = Input(shape=[criteo.get_categorical_count()])
        sparse_input_node = Input(shape=[criteo.get_categorical_count()])
        input = [dense_input_node, sparse_input_node]

        dense_feat_emb = DenseFeatureMapper(
            num_of_fields=criteo.get_numerical_count(),
            embedding_dim=64)(dense_input_node)

        sparse_feat_emb = SparseFeatureMapper(
            num_of_fields=criteo.get_categorical_count(),
            hash_size=criteo.get_hash_size(),
            embedding_dim=64)(sparse_input_node)

        emb_dict = {'dense': dense_feat_emb, 'sparse': sparse_feat_emb}


    # select model
    if args.model == 'dlrm':
        output = build_dlrm(emb_dict)
    if args.model == 'deepfm':
        output = build_deepfm(emb_dict)
    if args.model == 'crossnet':
        output = build_neumf(emb_dict)
    if args.model == 'autoint':
        output = build_autorec(emb_dict)
    # if args.model == 'neumf':
    #     output = build_autorec(emb_dict)
    if args.model == 'autorec':
        output = build_autorec(emb_dict)

    output = PointWiseOptimizer()(output)
    model = CTRRecommender(inputs=input, outputs=output)

    # search and predict.
    searcher = Search(model=model,
                      tuner=args.search,
                      tuner_params={'max_trials': args.trials, 'overwrite': True}
                      )
    start_time = time.time()
    searcher.search(x=train_X,
                    y=train_y,
                    x_val=val_X,
                    y_val=val_y,
                    objective='val_BinaryCrossentropy',
                    batch_size=args.batch_size,
                    epochs = 1,
                    callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)] 
                    )
    end_time = time.time()
    print( "running time:", end_time - start_time )
    print( "args", args)
    logger.info('First 10 Predicted Ratings: {}'.format(searcher.predict(x=val_X[:10])))
    logger.info('Predicting Accuracy (mse): {}'.format(searcher.evaluate(x=test_X, y_true=test_y)))
