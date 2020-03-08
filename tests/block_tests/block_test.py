import pytest
import unittest
import tensorflow as tf
from autorecsys.pipeline.interactor import MLPInteraction, ConcatenateInteraction, ElementwiseInteraction, FMInteraction
from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.pipeline.mapper import LatentFactorMapper, SparseFeatureMapper, DenseFeatureMapper
from autorecsys.pipeline.optimizer import RatingPredictionOptimizer, PointWiseOptimizer


class TestBlocks(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_blocks.ini").write("# testdata")

    def setUp(self):
        super(TestBlocks, self).setUp()
        self.inputs = [tf.constant([1, 2, 3], shape=(1, 3)), tf.constant([4, 5, 6], shape=(1, 3))]

    def test_MLPInteraction(self):
        mlp = MLPInteraction(units=128)
        hp = hp_module.HyperParameters()
        ans = mlp.build(hp, self.inputs)
        assert ans.shape == (1, 128)

    def test_ConcatenateInteraction(self):
        """
        Test class ConcatenateInteraction in interactor.py
        """
        sol = tf.constant([1, 2, 3, 4, 5, 6])
        hp = hp_module.HyperParameters()
        interactor = ConcatenateInteraction()
        ans = interactor.build(hp, self.inputs)
        assert tf.reduce_all(tf.equal(ans, sol))

    def test_ElementwiseInteraction(self):
        """
        Test class ElementwiseAddInteraction in interactor.py
        """
        sol_sum = tf.constant([5, 7, 9])
        sol_average = tf.constant([2.5, 3.5, 4.5])
        sol_max = tf.constant([4, 5, 6])
        sol_min = tf.constant([1, 2, 3])
        sol_innerporduct = tf.constant([4, 10, 18])
        hp = hp_module.HyperParameters()
        interactor = ElementwiseInteraction(elementwise_type='sum')
        ans = interactor.build(hp, self.inputs)
        assert tf.reduce_all(tf.equal(ans, sol_sum))

        interactor = ElementwiseInteraction(elementwise_type='average')
        ans = interactor.build(hp, [tf.cast(const, 'float') for const in self.inputs])
        assert tf.reduce_all(tf.equal(ans, sol_average))

        interactor = ElementwiseInteraction(elementwise_type='max')
        ans = interactor.build(hp, self.inputs)
        assert tf.reduce_all(tf.equal(ans, sol_max))

        interactor = ElementwiseInteraction(elementwise_type='min')
        ans = interactor.build(hp, self.inputs)
        assert tf.reduce_all(tf.equal(ans, sol_min))

        interactor = ElementwiseInteraction(elementwise_type='innerporduct')
        ans = interactor.build(hp, self.inputs)
        assert tf.reduce_all(tf.equal(ans, sol_innerporduct))

    def test_FMInteraction(self):
        inputs = tf.random.uniform(shape=[10, 10, 3])
        hp = hp_module.HyperParameters()
        interactor = FMInteraction()
        assert interactor.tunable_candidates == ['embedding_dim']
        ans = interactor.build(hp, inputs)
        assert ans.shape == (10, 1)

    def test_LatentFactorMapper(self):
        inputs = [tf.reshape(tf.range(12), (3, 4))]
        mapper = LatentFactorMapper(feat_column_id=0, id_num=10)
        hp = hp_module.HyperParameters()

        ans = mapper.build(hp, inputs)
        assert ans.shape == (3, 8)

    def test_DenseFeatureMapper(self):
        inputs = [tf.cast(tf.reshape(tf.range(12), (3, 4)), 'float')]
        mapper = DenseFeatureMapper(num_of_fields=4)
        hp = hp_module.HyperParameters()
        ans = mapper.build(hp, inputs)
        assert ans.shape == (3, 4, 8)

    def test_SparseFeatureMapper(self):
        inputs = [tf.cast(tf.reshape(tf.range(12), (3, 4)), 'float')]
        mapper = SparseFeatureMapper(num_of_fields=4)
        hp = hp_module.HyperParameters()
        ans = mapper.build(hp, inputs)
        assert ans.shape == (3, 4, 8)

    def test_RatingPredictionOptimizer(self):
        inputs = [tf.cast(tf.reshape(tf.range(12), (3, 4)), 'float')]
        opt = RatingPredictionOptimizer()
        hp = hp_module.HyperParameters()
        ans = opt.build(hp, inputs)
        assert ans.shape == (3, )
        assert isinstance(opt.metric, tf.keras.metrics.MeanSquaredError)
        assert isinstance(opt.loss, tf.losses.MeanSquaredError)

    def test_PointWiseOptimizer(self):
        inputs = [tf.cast(tf.reshape(tf.range(12), (3, 4)), 'float')]
        opt = PointWiseOptimizer()
        hp = hp_module.HyperParameters()
        ans = opt.build(hp, inputs)
        assert ans.shape == (3, )
        assert isinstance(opt.metric, tf.keras.metrics.BinaryCrossentropy)
        assert isinstance(opt.loss, tf.keras.losses.BinaryCrossentropy)
