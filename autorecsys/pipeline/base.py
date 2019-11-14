import logging
import time
import random
import json
import tempfile

from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.searcher.core.trial import Stateful
from tensorflow.python.util import nest


class Node(Stateful):
    """The nodes in a network connecting the blocks."""

    def __init__(self, shape=None):
        super().__init__()
        self.in_blocks = []
        self.out_blocks = []
        self.shape = shape

    def add_in_block(self, hypermodel):
        self.in_blocks.append(hypermodel)

    def add_out_block(self, hypermodel):
        self.out_blocks.append(hypermodel)

    def get_state(self):
        return {'shape': self.shape}

    def set_state(self, state):
        self.shape = state['shape']


class Block(Stateful):
    def __init__(self, name=None, tunable=True, directory=None):
        self.fixed_params = None
        self.tunable_candidates = None
        self.name = name or self._generate_random_name()
        self.tunable = tunable
        self.inputs = None
        self.outputs = None
        self._num_output_node = 1
        self._hyperparameters = None
        self.logger = logging.getLogger(self.name)
        self.directory = directory

    def _generate_random_name(self):
        prefix = self.__class__.__name__
        s = str(time.time()) + str(random.randint(1, 1e7))
        s = hash(s) % 1045543567
        name = prefix + '_' + str(s)
        return name

    def __str__(self):
        return self.name

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def __call__(self, inputs):
        """Functional API.
        # Arguments
            inputs: A list of input node(s) or a single input node for the block.
        # Returns
            list: A list of output node(s) of the Block.
        """
        inputs = nest.flatten(inputs)
        self.inputs = inputs
        for input_node in self.inputs:
            if not isinstance(input_node, Node):
                raise TypeError('Expect the inputs to layer {name} to be '
                                'a Node, but got {type}.'.format(
                    name=self.name,
                    type=type(input_node)))
            input_node.add_out_block(self)
        self.outputs = []
        for _ in range(self._num_output_node):
            output_node = Node()
            output_node.add_in_block(self)
            self.outputs.append(output_node)
        return self.outputs

    def get_state(self):
        """Get the configuration of the preprocessor.
        # Returns
            A dictionary of configurations of the preprocessor.
        """
        return {'name': self.name}

    def set_state(self, state):
        """Set the configuration of the preprocessor.
        # Arguments
            state: A dictionary of the configurations of the preprocessor.
        """
        self.name = state['name']

    def _check_fixed(self):
        if self.fixed_params is None:
            raise TypeError('fixed_params can not be None, it should be a list of parameters which are fixed')
        for fixed_param_name in self.fixed_params:
            param = getattr(self, fixed_param_name)
            if (not isinstance(param, hp_module.Fixed)) and (isinstance(param, hp_module.HyperParameter)):
                raise ValueError(f'{fixed_param_name} can not be set as hyper-parameters in the '
                                 f'{self.__class__.__name__}, it must be fixed')

    def _get_hyperparameters(self):
        if self.tunable_candidates is None or self.fixed_params is None:
            raise TypeError('tunable_candidates and fixed_params can not be None, it should '
                            'be a list of parameters which are tunable')
        hyperparameters = {}
        for param_name in self.tunable_candidates + self.fixed_params:
            param_val = getattr(self, param_name)
            if not isinstance(param_val, hp_module.HyperParameter):
                param_val = hp_module.Fixed(param_name, param_val)
            hyperparameters[param_name] = param_val
        return hyperparameters


class HyperBlock(Block):
    """HyperBlock uses hyperparameters to decide inner Block graph.
    A HyperBlock should be build into connected Blocks instead of individual Keras
    layers. The main purpose of creating the HyperBlock class is for the ease of
    parsing the graph for preprocessors. The graph would be hard to parse if a Block,
    whose inner structure is decided by hyperparameters dynamically, contains both
    preprocessors and Keras layers.
    When the preprocessing layers of Keras are ready to cover all the preprocessors
    in AutoKeras, the preprocessors should be handled by the Keras Model. The
    HyperBlock class should be removed. The subclasses should extend Block class
    directly and the build function should build connected Keras layers instead of
    Blocks.
    # Arguments
        output_shape: Tuple of int(s). Defaults to None. If None, the output shape
            will be inferred from the AutoModel.
        name: String. The name of the block. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def build(self, hp, inputs=None):
        """Build the HyperModel instead of Keras Model.
        # Arguments
            hp: HyperParameters. The hyperparameters for building the model.
            inputs: A list of instances of Node.
        # Returns
            An Node instance, the output node of the output Block.
        """
        raise NotImplementedError


class GeneralBlock(Block):
    """Hyper General block base class.
    It extends Block which extends Hypermodel. A GeneralBlock is a Hypermodel, which
    means it is a search space. However, different from other Hypermodels, it is
    also a model whose config could be set.
    """

    def get_config(self):
        """Get the configuration of the preprocessor.
        # Returns
            A dictionary of configurations of the preprocessor.
        """
        return {}

    def set_config(self, config):
        """Set the configuration of the preprocessor.
        # Arguments
            config: A dictionary of the configurations of the preprocessor.
        """
        pass

    def get_weights(self):
        """Get the trained weights of the preprocessor.
        # Returns
            A dictionary of trained weights of the preprocessor.
        """
        return {}

    def set_weights(self, weights):
        """Set the trained weights of the preprocessor.
        # Arguments
            weights: A dictionary of trained weights of the preprocessor.
        """
        pass

    def get_state(self):
        state = super().get_state()
        state.update(self.get_config())
        return {'config': state,
                'weights': self.get_weights()}

    def set_state(self, state):
        self.set_config(state['config'])
        super().set_state(state['config'])
        self.set_weights(state['weights'])

    def initialize(self):
        pass

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        with open(filename, 'r') as fp:
            weights = json.load(fp)
        self.set_weights(weights)


class Preprocessor(GeneralBlock):
    """Hyper preprocessing block base class.
    It extends Block which extends GeneralBlock. A preprocessor is a Hypermodel, which
    means it is a search space. However, different from other Hypermodels and GeneralBlock, it is
    also a model which can be fit.
    """

    def fit_transform(self, x, y):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.
            fit: Boolean. Whether it is in fit mode.

        Returns:
            A transformed instanced which can be converted to a tf.Tensor.
        """
        raise NotImplementedError

    def transform(self, x):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.
            fit: Boolean. Whether it is in fit mode.

        Returns:
            A transformed instanced which can be converted to a tf.Tensor.
        """
        raise NotImplementedError
