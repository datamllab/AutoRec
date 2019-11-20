import pytest
import tensorflow as tf
from autorecsys.searcher.core import hyperparameters as hp_module

from autorecsys.pipeline import Input 
from autorecsys.pipeline import graph as graph_module


# def test_set_hp():
#     input_node = Input((32,))
#     output_node = input_node
#     output_node = ak.DenseBlock()(output_node)
#     head = ak.RegressionHead()
#     head.output_shape = (1,)
#     output_node = head(output_node)

#     graph = graph_module.HyperGraph(
#         input_node,
#         output_node,
#         override_hps=[hp_module.Choice('dense_block_1/num_layers', [6], default=6)])
#     hp = hp_module.HyperParameters()
#     plain_graph = graph.hyper_build(hp)
#     plain_graph.build_keras_graph().build(hp)

#     for single_hp in hp.space:
#         if single_hp.name == 'dense_block_1/num_layers':
#             assert len(single_hp.values) == 1
#             assert single_hp.values[0] == 6
#             return
#     assert False


# def test_input_output_disconnect():
#     input_node1 = ak.Input()
#     output_node = input_node1
#     _ = ak.DenseBlock()(output_node)

#     input_node = ak.Input()
#     output_node = input_node
#     output_node = ak.DenseBlock()(output_node)
#     output_node = ak.RegressionHead()(output_node)

#     with pytest.raises(ValueError) as info:
#         graph_module.HyperGraph(input_node1, output_node)
#     assert 'Inputs and outputs not connected.' in str(info.value)


# def test_hyper_graph_cycle():
#     input_node1 = ak.Input()
#     input_node2 = ak.Input()
#     output_node1 = ak.DenseBlock()(input_node1)
#     output_node2 = ak.DenseBlock()(input_node2)
#     output_node = ak.Merge()([output_node1, output_node2])
#     head = ak.RegressionHead()
#     output_node = head(output_node)
#     head.outputs = output_node1

#     with pytest.raises(ValueError) as info:
#         graph_module.HyperGraph([input_node1, input_node2], output_node)
#     assert 'The network has a cycle.' in str(info.value)


# def test_input_missing():
#     input_node1 = ak.Input()
#     input_node2 = ak.Input()
#     output_node1 = ak.DenseBlock()(input_node1)
#     output_node2 = ak.DenseBlock()(input_node2)
#     output_node = ak.Merge()([output_node1, output_node2])
#     output_node = ak.RegressionHead()(output_node)

#     with pytest.raises(ValueError) as info:
#         graph_module.HyperGraph(input_node1, output_node)
#     assert 'A required input is missing for HyperModel' in str(info.value)


# def test_graph_basics():
#     input_node = ak.Input(shape=(30,))
#     output_node = input_node
#     output_node = ak.DenseBlock()(output_node)
#     output_node = ak.RegressionHead(output_shape=(1,))(output_node)

#     graph = graph_module.PlainGraph(input_node, output_node)
#     model = graph.build_keras_graph().build(hp_module.HyperParameters())
#     assert model.input_shape == (None, 30)
#     assert model.output_shape == (None, 1)


# def test_merge():
#     input_node1 = ak.Input(shape=(30,))
#     input_node2 = ak.Input(shape=(40,))
#     output_node1 = ak.DenseBlock()(input_node1)
#     output_node2 = ak.DenseBlock()(input_node2)
#     output_node = ak.Merge()([output_node1, output_node2])
#     output_node = ak.RegressionHead(output_shape=(1,))(output_node)

#     graph = graph_module.PlainGraph([input_node1, input_node2],
#                                     output_node)
#     model = graph.build_keras_graph().build(hp_module.HyperParameters())
#     assert model.input_shape == [(None, 30), (None, 40)]
#     assert model.output_shape == (None, 1)
