from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import tensorflow as tf
from autorecsys.pipeline.interactor import (
    RandomSelectInteraction,
    ConcatenateInteraction,
    InnerProductInteraction,
    ElementwiseInteraction,
    MLPInteraction,
    HyperInteraction,
    FMInteraction,
    CrossNetInteraction,
    SelfAttentionInteraction,
)
from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.utils.common import set_seed


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU

logger = logging.getLogger(__name__)


class TestInteractors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_interactor.ini").write("# testdata")

    def setUp(self):
        super(TestInteractors, self).setUp()
        # Wrap tensor in an extra list to accommodate the batch dimension
        self.inputs = [tf.constant([[1, 2, 3]], dtype='float32'), tf.constant([[4, 5, 6]], dtype='float32')]

    def test_RandomSelectInteraction(self):
        # TODO: Sean
        # Step 1: Test constructor and get_state
        p = {}
        interactor = RandomSelectInteraction(**p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'random_select_interaction_1',
        }
        assert ans_state == sol_state

        # Step 2: Test set_state
        p = {}
        interactor.set_state(p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'random_select_interaction_1',
        }
        assert ans_state == sol_state

        # Step 3: Test build and associated functions
        reps = 100  # Arrange
        sol = [tf.constant([[1, 2, 3]], dtype='float32')] * reps  # extra list compensates for batch dimension
        ans = list()
        hp = hp_module.HyperParameters()
        interactor = RandomSelectInteraction()
        for _ in range(reps):
            set_seed()
            ans.append(interactor.build(hp, self.inputs))  # Act
        assert all([all(tf.equal(a, s)[0]) for a, s in zip(ans, sol)])  # Assert: [0] unwraps the batch dimension

    def test_ConcatenateInteraction(self):
        """
        Test class ConcatenateInteraction in interactor.py
        """

        hp = hp_module.HyperParameters()
        interactor = ConcatenateInteraction()
        output = interactor.build(hp, self.inputs)  # Act

        # output
        assert len(tf.nest.flatten(output)) == 1

        sol = tf.constant([[1, 2, 3, 4, 5, 6]], dtype=tf.float32)  # Arrange
        assert tf.reduce_all(tf.equal(output, sol))  # Assert

    def test_MLPInteraction(self):
        """
        Test class MLPInteraction in interactor.py
        """
        # intialize
        hp = hp_module.HyperParameters()
        p = {
            'units': 16,
            'num_layers': 2,
            'use_batchnorm': False,
            'dropout_rate': 0.25}  # units, num_layer, use_batchnorm, dropout
        interactor = MLPInteraction(**p)

        # test get_state()
        sol_get_state = {
            'name': 'mlp_interaction_1',
            'units': 16,
            'num_layers': 2,
            'use_batchnorm': False,
            'dropout_rate': 0.25}

        assert interactor.get_state() == sol_get_state

        # test set_state()
        p = {
            'units': 32,
            'num_layers': 1,
            'use_batchnorm': True,
            'dropout_rate': 0.0}
        sol_set_state = {
            'name': 'mlp_interaction_1',
            'units': 32,
            'num_layers': 1,
            'use_batchnorm': True,
            'dropout_rate': 0.0}
        interactor.set_state(p)
        ans_set_state = interactor.get_state()
        assert ans_set_state == sol_set_state

        # output shape
        output = interactor.build(hp, self.inputs)  # Act
        assert len(tf.nest.flatten(output)) == 1

    def test_FMInteraction(self):
        """
        Test class FMInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = {
            'embedding_dim': 8}  # embedding_dim
        interactor = FMInteraction(**p)

        # test get_state()
        sol_get_state = {
            'name': 'fm_interaction_1',
            'embedding_dim': 8}

        assert interactor.get_state() == sol_get_state

        # test set_state()
        p = {
            'embedding_dim': 16}
        sol_set_state = {
            'name': 'fm_interaction_1',
            'embedding_dim': 16}
        interactor.set_state(p)
        ans_set_state = interactor.get_state()
        assert ans_set_state == sol_set_state

        # output shape
        output = interactor.build(hp, self.inputs)  # Act
        assert len(tf.nest.flatten(output)) == 1

    def test_CrossNetInteraction(self):
        """
        Test class CrossNetInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = {
            'layer_num': 1}  # embedding_dim
        interactor = CrossNetInteraction(**p)

        # test get_state()
        sol_get_state = {
            'name': 'cross_net_interaction_1',
            'layer_num': 1}

        assert interactor.get_state() == sol_get_state

        # test set_state()
        p = {
            'layer_num': 2}
        sol_set_state = {
            'name': 'cross_net_interaction_1',
            'layer_num': 2}
        interactor.set_state(p)
        ans_set_state = interactor.get_state()
        assert ans_set_state == sol_set_state

        # output shape
        output = interactor.build(hp, self.inputs)  # Act
        assert len(tf.nest.flatten(output)) == 1
        # Step 1: Test constructor and get_state
        p = {}
        interactor = ConcatenateInteraction(**p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'concatenate_interaction_1',
        }
        assert ans_state == sol_state

        # Step 2: Test set_state
        p = {}
        interactor.set_state(p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'concatenate_interaction_1',
        }
        assert ans_state == sol_state

        # Step 3: Test build and associated functions
        sol = tf.constant([[1, 2, 3, 4, 5, 6]], dtype='float32')  # Arrange
        hp = hp_module.HyperParameters()
        interactor = ConcatenateInteraction()
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

    def test_InnerProductInteraction(self):
        """
        Test class InnerProductInteraction in interactor.py
        """
        # Step 1: Test constructor and get_state
        p = {}
        interactor = InnerProductInteraction(**p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'inner_product_interaction_1',
        }
        assert ans_state == sol_state

        # Step 2: Test set_state
        p = {}
        interactor.set_state(p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'inner_product_interaction_1',
        }
        assert ans_state == sol_state

        # Step 3: Test build and associated functions
        sol = tf.constant([[32]], dtype='float32')  # Arrange
        hp = hp_module.HyperParameters()
        interactor = InnerProductInteraction()
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

    def test_ElementwiseInteraction(self):
        # TODO: Sean
        # Step 1: Test constructor and get_state
        p = {
            'elementwise_type': 'average',
        }
        interactor = ElementwiseInteraction(**p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'elementwise_interaction_1',
            'elementwise_type': 'average',
        }
        print(ans_state)
        assert ans_state == sol_state

        # Step 2: Test set_state
        p = {
            'elementwise_type': 'multiply',
        }
        interactor.set_state(p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'elementwise_interaction_1',
            'elementwise_type': 'multiply',
        }
        assert ans_state == sol_state

        # Step 3: Test build and associated functions
        # Step 3.1: Test elementwise sum
        hp = hp_module.HyperParameters()  # Arrange
        sol = tf.constant([[5, 7, 9]], dtype='float32')
        interactor = ElementwiseInteraction('sum')  # Arrange
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

        # Step 3.2: Test elementwise average
        sol = tf.constant([[2.5, 3.5, 4.5]], dtype='float32')  # Arrange
        interactor = ElementwiseInteraction('average')
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

        # Step 3.3: Test elementwise multiply (Hadamard product)
        sol = tf.constant([[4, 10, 18]], dtype='float32')  # Arrange
        interactor = ElementwiseInteraction('multiply')
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

        # Step 3.4: Test elementwise max
        sol = tf.constant([[4, 5, 6]], dtype='float32')  # Arrange
        interactor = ElementwiseInteraction('max')
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

        # Step 3.5: Test elementwise min
        sol = tf.constant([[1, 2, 3]], dtype='float32')  # Arrange
        interactor = ElementwiseInteraction('min')
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol)[0])  # Assert

    def test_MLPInteraction(self):
        # TODO: Suil
        pass

    def test_HyperInteraction(self):
        # TODO: Sean
        # Step 1: Test constructor and get_state
        p = {
            'meta_interator_num': 3,
            'interactor_type': 'ConcatenateInteraction',
        }
        interactor = HyperInteraction(**p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'hyper_interaction_1',
            'meta_interator_num': 3,
            'interactor_type': 'ConcatenateInteraction',
            'name2interactor': {
                'RandomSelectInteraction': RandomSelectInteraction,
                'ConcatenateInteraction': ConcatenateInteraction,
                'InnerProductInteraction': InnerProductInteraction,
                'ElementwiseInteraction': ElementwiseInteraction,
                'MLPInteraction': MLPInteraction,
                'FMInteraction': FMInteraction,
                'CrossNetInteraction': CrossNetInteraction,
                'SelfAttentionInteraction': SelfAttentionInteraction,
            }
        }
        assert ans_state == sol_state

        # Step 2: Test set_state
        p = {
            'meta_interator_num': 6,
            'interactor_type': 'MLPInteraction',
        }
        interactor.set_state(p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'hyper_interaction_1',
            'meta_interator_num': 6,
            'interactor_type': 'MLPInteraction',
            'name2interactor': {
                'RandomSelectInteraction': RandomSelectInteraction,
                'ConcatenateInteraction': ConcatenateInteraction,
                'InnerProductInteraction': InnerProductInteraction,
                'ElementwiseInteraction': ElementwiseInteraction,
                'MLPInteraction': MLPInteraction,
                'FMInteraction': FMInteraction,
                'CrossNetInteraction': CrossNetInteraction,
                'SelfAttentionInteraction': SelfAttentionInteraction,
            }
        }
        assert ans_state == sol_state

        # Step 3: Test build and associated functions
        hp = hp_module.HyperParameters()
        ans = interactor.build(hp, self.inputs)  # Act
        sol = 1
        assert len(tf.nest.flatten(ans)) == sol

    def test_FMInteraction(self):
        # TODO: Suil
        pass

    def test_CrossNetInteraction(self):
        # TODO: Suil
        pass

    def test_SelfAttentionInteraction(self):
        # TODO: Sean
        # Step 1: Test constructor and get_state
        p = {
            'embedding_dim': 8,
            'att_embedding_dim': 8,
            'head_num': 2,
            'residual': True,
        }
        interactor = SelfAttentionInteraction(**p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'self_attention_interaction_1',
            'embedding_dim': 8,
            'att_embedding_dim': 8,
            'head_num': 2,
            'residual': True,
        }
        assert ans_state == sol_state

        # Step 2: Test set_state
        p = {
            'embedding_dim': 16,
            'att_embedding_dim': 16,
            'head_num': 4,
            'residual': False,
        }
        interactor.set_state(p)
        ans_state = interactor.get_state()
        sol_state = {
            'name': 'self_attention_interaction_1',
            'embedding_dim': 16,
            'att_embedding_dim': 16,
            'head_num': 4,
            'residual': False,
        }
        assert ans_state == sol_state

        # Step 3: Test build and associated functions
        hp = hp_module.HyperParameters()
        ans = interactor.build(hp, self.inputs)  # Act
        sol = 1
        assert len(tf.nest.flatten(ans)) == sol
