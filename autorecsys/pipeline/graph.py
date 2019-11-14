import pickle

from autokaggle.base.trial import Stateful
import tensorflow as tf
from tensorflow.python.util import nest


class Graph(Stateful):
    """A graph consists of connected Blocks, HyperBlocks, Preprocessors or Heads.

    # Arguments
        inputs: A list of input node(s) for the Graph.
        outputs: A list of output node(s) for the Graph.
        override_hps: A list of HyperParameters. The predefined HyperParameters that
            will override the space of the Hyperparameters defined in the Hypermodels
            with the same names.
    """

    def __init__(self, inputs, outputs):
        super().__init__()
        # TODO flatten inputs & outputs
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        # reverse order of the topological sort
        self._node_to_id = {}
        self._nodes = []
        # topological sort of the blocks in the graph
        self._blocks = []
        self._block_to_id = {}
        self._build_network()

    def compile(self, func):
        """Share the information between blocks by calling functions in compiler.

        # Arguments
            func: A dictionary. The keys are the block classes. The values are
                corresponding compile functions.
        """
        for block in self._blocks:
            if block.__class__ in func:
                func[block.__class__](block)

    def _build_network(self):
        self._node_to_id = {}

        # Recursively find all the interested nodes.
        for input_node in self.inputs:
            self._search_network(input_node, self.outputs, set(), set())
        # the topological sort of the graph in reverse order
        self._nodes = sorted(list(self._node_to_id.keys()),
                             key=lambda x: self._node_to_id[x])

        for node in (self.inputs + self.outputs):
            if node not in self._node_to_id:
                raise ValueError('Inputs and outputs not connected.')

        # Find the blocks.
        blocks = []
        for input_node in self._nodes:
            for block in input_node.out_blocks:
                if any([output_node in self._node_to_id
                        for output_node in block.outputs]) and block not in blocks:
                    blocks.append(block)

        # Check if all the inputs of the blocks are set as inputs.
        for block in blocks:
            for input_node in block.inputs:
                if input_node not in self._node_to_id:
                    raise ValueError('A required input is missing for HyperModel '
                                     '{name}.'.format(name=block.name))

        # Calculate the in degree of all the nodes
        in_degree = [0] * len(self._nodes)
        for node_id, node in enumerate(self._nodes):
            in_degree[node_id] = len([
                block for block in node.in_blocks if block in blocks])

        # Add the blocks in topological order.
        self._blocks = []
        self._block_to_id = {}
        while len(blocks) != 0:
            new_added = []

            # Collect blocks with in degree 0.
            for block in blocks:
                if any([in_degree[self._node_to_id[node]]
                        for node in block.inputs]):
                    continue
                new_added.append(block)

            # Remove the collected blocks from blocks.
            for block in new_added:
                blocks.remove(block)

            for block in new_added:
                # Add the collected blocks to the AutoModel.
                self._add_block(block)

                # Decrease the in degree of the output nodes.
                for output_node in block.outputs:
                    if output_node not in self._node_to_id:
                        continue
                    output_node_id = self._node_to_id[output_node]
                    in_degree[output_node_id] -= 1

    def _search_network(self, input_node, outputs, in_stack_nodes,
                        visited_nodes):
        visited_nodes.add(input_node)
        in_stack_nodes.add(input_node)

        outputs_reached = False
        if input_node in outputs:
            outputs_reached = True

        for block in input_node.out_blocks:
            for output_node in block.outputs:
                if output_node in in_stack_nodes:
                    raise ValueError('The network has a cycle.')
                if output_node not in visited_nodes:
                    self._search_network(output_node, outputs, in_stack_nodes,
                                         visited_nodes)
                if output_node in self._node_to_id.keys():
                    outputs_reached = True

        if outputs_reached:
            self._add_node(input_node)

        in_stack_nodes.remove(input_node)

    def _add_block(self, block):
        if block not in self._blocks:
            block_id = len(self._blocks)
            self._block_to_id[block] = block_id
            self._blocks.append(block)

    def _add_node(self, input_node):
        if input_node not in self._node_to_id:
            self._node_to_id[input_node] = len(self._node_to_id)

    def _get_block(self, name):
        for block in self._blocks:
            if block.name == name:
                return block
        raise ValueError('Cannot find block named {name}.'.format(name=name))

    def get_state(self):
        block_state = {str(block_id): block.get_state()
                       for block_id, block in enumerate(self._blocks)}
        node_state = {str(node_id): node.get_state()
                      for node_id, node in enumerate(self._nodes)}
        return {'blocks': block_state, 'nodes': node_state}

    def set_state(self, state):
        block_state = state['blocks']
        node_state = state['nodes']
        for block_id, block in enumerate(self._blocks):
            block.set_state(block_state[str(block_id)])
        for node_id, node in enumerate(self._nodes):
            node.set_state(node_state[str(node_id)])

    def save(self, fname):
        state = self.get_state()
        with open(fname, 'w') as f:
            pickle.dump(state, f)
        return str(fname)

    def reload(self, fname):
        with open(fname, 'r') as f:
            state = pickle.load(f)
        self.set_state(state)
