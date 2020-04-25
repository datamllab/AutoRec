from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.util import nest
from autorecsys.pipeline.base import Block
from autorecsys.pipeline.utils import Bias
import random
import tensorflow as tf
import numpy as np


class RandomSelectInteraction(Block):
    """Module for output one vector select form the input vector list .
    # Attributes:
        None
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_state(self):
        state = super().get_state()
        return state

    def set_state(self, state):
        super().set_state(state)

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)
        output_node = random.choice(input_node)
        return output_node


class ConcatenateInteraction(Block):
    """Module for outputing one vector by concatenating the input vector list.
    # Attributes:
        None
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_state(self):
        state = super().get_state()
        return state

    def set_state(self, state):
        super().set_state(state)

    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        output_node = tf.concat(input_node, axis=1)
        return output_node


class ElementwiseInteraction(Block):
    """Module for element-wise operation. this block includes the element-wise sum ,average, innerporduct, max, and min.
        The default operation is average.
    # Attributes:
        elementwise_type("str"):  Can be used to select the element-wise operation. the default value is None. If the
        value of this parameter is None, the block can select the operation for the
        sum ,average, innerporduct, max, and min, according to the search algorithm.
    """

    def __init__(self,
                 elementwise_type=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.elementwise_type = elementwise_type

    def get_state(self):
        state = super().get_state()
        state.update({
            'elementwise_type': self.elementwise_type})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.elementwise_type = state['elementwise_type']

    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]

        shape_set = set()
        for node in input_node:
            shape_set.add(node.shape[1])
        if len(shape_set) > 1:
            # raise ValueError("Inputs of ElementwiseInteraction should have same dimension.")
            min_len = min( shape_set )

            input_node = [tf.keras.layers.Dense(min_len)(node) 
                        if node.shape[1] != min_len  else node for node in input_node]

        elementwise_type = self.elementwise_type or hp.Choice('elementwise_type',
                                                              ["sum", "average", "innerporduct", "max", "min"],
                                                              default='average')
        if elementwise_type == "sum":
            output_node = tf.add_n(input_node)
        elif elementwise_type == "average":
            output_node = tf.reduce_mean(input_node, axis=0)
        elif elementwise_type == "innerporduct":
            output_node = tf.reduce_prod(input_node, axis=0)
        elif elementwise_type == "max":
            output_node = tf.reduce_max(input_node, axis=[0])
        elif elementwise_type == "min":
            output_node = tf.reduce_min(input_node, axis=[0])
        else:
            output_node = tf.add_n(input_node)
        return output_node


class MLPInteraction(Block):
    """Module for MLP operation. This block can seted as MLP with different layer, unit, and other setting .
    # Attributes:
        units (int). The units of all layer in the MLP block.
        num_layers (int). The number of the layers in the MLP blck
        use_batchnorm (Boolean). Use batch normalization or not.
        dropout_rate(float). The value of drop out in the last layer of MLP.
    """

    def __init__(self,
                 units=None,
                 num_layers=None,
                 use_batchnorm=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

    def get_state(self):
        state = super().get_state()
        state.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.units = state['units']
        self.num_layers = state['num_layers']
        self.use_batchnorm = state['use_batchnorm']
        self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        output_node = tf.concat(input_node, axis=1)
        num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Choice('use_batchnorm', [True, False], default=False)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        for i in range(num_layers):
            units = self.units or hp.Choice(
                'units_{i}'.format(i=i),
                [16, 32, 64, 128, 256, 512, 1024],
                default=32)
            output_node = tf.keras.layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = tf.keras.layers.BatchNormalization()(output_node)
            output_node = tf.keras.layers.ReLU()(output_node)
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node


class HyperInteraction(Block):
    """Module for selecting different block. This block includes can select different blocks in the interactor.
    # Attributes:
        meta_interator_num (str). The total number of the meta interoctor block.
        interactor_type (str).  The type of interactor used in this block.
    """

    def __init__(self, meta_interator_num=None, interactor_type=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_interator_num = meta_interator_num
        self.interactor_type = interactor_type
        self.name2interactor = {
            "MLPInteraction": MLPInteraction,
            "ConcatenateInteraction": ConcatenateInteraction,
            "RandomSelectInteraction": RandomSelectInteraction,
            "ElementwiseInteraction": ElementwiseInteraction,
            "FMInteraction": FMInteraction, 
            "CrossNetInteraction": CrossNetInteraction,
            "SelfAttentionInteraction": SelfAttentionInteraction,
        }

    def get_state(self):
        state = super().get_state()
        state.update({
            'interactor_type': self.interactor_type,
            'meta_interator_num': self.meta_interator_num
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.interactor_type = state['interactor_type']
        self.meta_interator_num = state['meta_interator_num']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)
        meta_interator_num = self.meta_interator_num or hp.Choice('meta_interator_num',
                                                                  [1, 2, 3, 4, 5, 6],
                                                                  default=3)
        interactors_name = []
        for idx in range(meta_interator_num):
            tmp_interactor_type = self.interactor_type or hp.Choice('interactor_type_' + str(idx),
                                                                    list(self.name2interactor.keys()),
                                                                    default='ConcatenateInteraction')
            interactors_name.append(tmp_interactor_type)

        outputs = [self.name2interactor[interactor_name]().build(hp, input_node) 
                                for interactor_name in interactors_name]

        # DO WE REALLY NEED TO CAT THEM?
        outputs = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in outputs]
        outputs = tf.concat(outputs, axis=1)
        return outputs


class FMInteraction(Block):
    """CTR module for factorization machine operation.

    Reference: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

    This block applies factorization machine operation on a list of 
    input 3D tensors of size (batch_size, field_size, embedding_size). 
    It will align the dimension of tensors to 3D if they're 1D or 2D originally, and 
    will align/transfrom the last embedding dimension based on a tunable hyperaparmeter.

    # Attributes:
        embedding_dim (int). The transformed embedding dimension of each field,
                            before conducting the factorization machine operation.
    """

    def __init__(self,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = []
        self.tunable_candidates = ['embedding_dim']
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embedding_dim': self.embedding_dim,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        # expland all the tensors to 3D tensor 
        for idx, node in enumerate(input_node):
            if len(node.shape) == 1:
                input_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
            elif len(node.shape) == 2:
                input_node[idx] = tf.expand_dims(node, 1) 
            elif len(node.shape) > 3:
                raise ValueError(
                    "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape)
                )

        # align the embedding_dim of input nodes if they're not the same
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [4, 8, 16], default=8)
        output_node = [tf.keras.layers.Dense(embedding_dim)(node) 
                        if node.shape[2] != embedding_dim  else node for node in input_node]
        output_node = tf.concat(output_node, axis=1)

        square_of_sum = tf.square(tf.reduce_sum(output_node, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(output_node * output_node, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        output_node = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        return output_node




class CrossNetInteraction(Block):
    """CTR module for crossnet layer in deep & cross network.

    Reference: https://arxiv.org/pdf/1708.05123.pdf

    This block applies cross interaction operation on a 2D tensors of size 
    (batch_size, embedding_size). 

    We assume the input could be a list of tensors of 2D or 3D, and the block will 
    flatten them as as list of 2D tensors, and ten concatenate them as a single 2D
    tensor. The cross interaction follows the reference and the number of 
    layers of the cross interaction is tunable.


    # Attributes:
        layer_num (int). The number of layers of the cross interaction.
    """
    def __init__(self, 
                layer_num=None, 
                **kwargs):

        super().__init__(**kwargs)
        self.layer_num = layer_num

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'layer_num': self.layer_num,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.layer_num = state['layer_num']


    def build(self, hp, inputs=None):

        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        input_node = tf.concat(input_node, axis=1)

        layer_num = self.layer_num or hp.Choice('layer_num', [1, 2, 3, 4], default=1)
        embedding_dim = input_node.shape[-1]

        # perform the multilayer cross net interaction
        output_node = input_node
        for _ in range(layer_num):
            pre_output_emb = tf.keras.layers.Dense(1, use_bias=False)(output_node) 
            cross_dot = tf.math.multiply(input_node, pre_output_emb)
            output_node = cross_dot + output_node
            output_node = Bias(embedding_dim)(output_node)

        return output_node 



class SelfAttentionInteraction(Block):
    """CTR module for the multi-head self-attention layer in the autoint paper.

    Reference: https://arxiv.org/pdf/1708.05123.pdf

    This block applies multi-head self-attention on a 3D tensor of size 
    (batch_size, field_size, embedding_size). 
    
    We assume the input could be a list of tensors of 1D, 2D or 3D, and the block 
    will align the dimension of tensors to 3D if they're 1D or 2D originally, and 
    it will also align the last embedding dimension based on a tunable hyperaparmeter.

    # Attributes:
        embedding_dim (int). Embedding dimension for aligning embedding dimension of 
                            the input tensors.
        att_embedding_dim (int). Output embedding dimension after the mulit-head self-attention.
        head_num (int). Number of attention heads.
        residual (boolean). Whether to apply residual connection after self-attention or not.
    """
    def __init__(self, 
                  embedding_dim=None, 
                  att_embedding_dim=None, 
                  head_num=None, 
                  residual=None, 
                  **kwargs):
        super(SelfAttentionInteraction, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.att_embedding_dim = embedding_dim
        self.head_num = head_num
        self.residual = residual

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embedding_dim': self.embedding_dim,
                'att_embedding_dim': self.att_embedding_dim,
                'head_num': self.head_num,
                'residual': self.residual,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.embedding_dim = state['embedding_dim']
        self.att_embedding_dim = state['att_embedding_dim']
        self.head_num = state['head_num']
        self.residual = state['residual']

    def _scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights. 

        Reference: https://www.tensorflow.org/tutorials/text/transformer

        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.
        
        # Arguments:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          
        # Returns:
          single-head attention result
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True) 
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        # expland all the tensors to 3D tensor 
        for idx, node in enumerate(input_node):
            if len(node.shape) == 1:
                input_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
            elif len(node.shape) == 2:
                input_node[idx] = tf.expand_dims(node, 1) 
            elif len(node.shape) > 3:
                raise ValueError(
                    "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape)
                )

        # align the embedding_dim of input nodes if they're not the same
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [4, 8, 16], default=8)
        output_node = [tf.keras.layers.Dense(embedding_dim)(node) 
                        if node.shape[2] != embedding_dim  else node for node in input_node]
        output_node = tf.concat(output_node, axis=1)


        att_embedding_dim = self.att_embedding_dim or hp.Choice('att_embedding_dim', [4, 8, 16], default=8)
        head_num = self.head_num or hp.Choice('head_num', [1, 2, 3, 4], default=2)
        residual = self.residual or hp.Choice('residual', [True, False], default=True)

        outputs = []
        for _ in range(head_num):
            query = tf.keras.layers.Dense(att_embedding_dim, use_bias=False)(output_node) 
            key = tf.keras.layers.Dense(att_embedding_dim, use_bias=False)(output_node) 
            value = tf.keras.layers.Dense(att_embedding_dim, use_bias=False)(output_node) 
          
            outputs.append(
                            self._scaled_dot_product_attention(query, key, value)
                          )

        outputs = tf.concat(outputs, axis=2)

        if self.residual:
            outputs += tf.keras.layers.Dense(att_embedding_dim * head_num, use_bias=False)(output_node)
                  
        return output_node


