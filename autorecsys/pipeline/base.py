import types
import tensorflow as tf
from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.searcher.core.trial import Stateful
from autorecsys.utils.common import to_snake_case
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

    def build(self):
        return tf.keras.Input(shape=self.shape)

    def get_state(self):
        return {'shape': self.shape}

    def set_state(self, state):
        self.shape = state['shape']


class HyperModel(object):
    """Defines a searchable space of Models and builds Models from this space.
    # Attributes:
        name: The name of this HyperModel.
        tunable: Whether the hyperparameters defined in this hypermodel
          should be added to search space. If `False`, either the search
          space for these parameters must be defined in advance, or the
          default values will be used.
    """

    def __init__(self, name=None, tunable=True):
        self.name = name
        self.tunable = tunable

        self._build = self.build
        self.build = self._build_wrapper

    def build(self, hp):
        """Builds a model.
        # Arguments:
            hp: A `HyperParameters` instance.
        # Returns:
            A model instance.
        """
        raise NotImplementedError

    def _build_wrapper(self, hp, *args, **kwargs):
        if not self.tunable:
            # Copy `HyperParameters` object so that new entries are not added
            # to the search space.
            hp = hp.copy()
        return self._build(hp, *args, **kwargs)


class Block(HyperModel, Stateful):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = None
        self.tunable_candidates = None
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
            name = to_snake_case(name)
        self._hyperparameters = None
        self.name = name
        self.inputs = None
        self.outputs = None
        self._num_output_node = 1

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        build_fn = obj.build

        def build_wrapper(obj, hp, *args, **kwargs):
            with hp.name_scope(obj.name):
                return build_fn(hp, *args, **kwargs)

        obj.build = types.MethodType(build_wrapper, obj)
        return obj

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
        if 'name' in state:
            self.name = state['name']


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


class Preprocessor(Block):
    """Hyper preprocessing block base class.
    It extends Block which extends Hypermodel. A preprocessor is a Hypermodel, which
    means it is a search space. However, different from other Hypermodels, it is
    also a model which can be fit.
    """

    def build(self, hp):
        """Get the values of the required HyperParameters.
        It does not build and return a Keras Model, but initialize the
        HyperParameters for the preprocessor to be fit.
        """
        pass

    def update(self, x, y=None):
        """Incrementally fit the preprocessor with a single training instance.
        # Arguments
            x: EagerTensor. A single instance in the training dataset.
            y: EagerTensor. The targets of the tasks. Defaults to None.
        """
        raise NotImplementedError

    def transform(self, x, fit=False):
        """Incrementally fit the preprocessor with a single training instance.
        # Arguments
            x: EagerTensor. A single instance in the training dataset.
            fit: Boolean. Whether it is in fit mode.
        Returns:
            A transformed instanced which can be converted to a tf.Tensor.
        """
        raise NotImplementedError

    def output_types(self):
        """The output types of the transformed data, e.g. tf.int64.
        The output types are required by tf.py_function, which is used for transform
        the dataset into a new one with a map function.
        # Returns
            A tuple of data types.
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        """The output shape of the transformed data.
        The output shape is needed to build the Keras Model from the AutoModel.
        The output shape of the preprocessor is the input shape of the Keras Model.
        # Returns
            A tuple of int(s) or a TensorShape.
        """
        raise NotImplementedError

    def finalize(self):
        """Training process of the preprocessor after update with all instances."""
        pass

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
