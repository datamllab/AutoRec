import pytest
import tensorflow as tf
from autorecsys.searcher.core import hyperparameters as hp_module

from autorecsys.pipeline import Input, MLPInteraction, ConcatenateInteraction, RatingPredictionOptimizer 
from autorecsys.pipeline import graph as graph_module

# TODO: we don't support overwrite hp for graph now.
# def test_set_hp():
#     input_node = Input((32,))
#     output_node = input_node
#     output_node = MLPInteraction()(output_node)
#     output_node = RatingPredictionOptimizer()[output_node]

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


def test_input_output_disconnect():
    input_node1 = Input()
    output_node = input_node1
    _ = MLPInteraction()(output_node)

    input_node = Input()
    output_node = input_node
    output_node = MLPInteraction()(output_node)
    output_node = RatingPredictionOptimizer()(output_node)

    with pytest.raises(ValueError) as info:
        graph_module.HyperGraph(input_node1, output_node)
    assert 'Inputs and outputs not connected.' in str(info.value)


# def test_hyper_graph_cycle():
#     input_node1 = Input()
#     input_node2 = Input()
#     output_node1 = MLPInteraction()(input_node1)
#     output_node2 = MLPInteraction()(input_node2)
#     output_node = ConcatenateInteraction()([output_node1, output_node2])
#     head = RatingPredictionOptimizer()
#     output_node = head(output_node)
#     head.outputs = output_node1

#     with pytest.raises(ValueError) as info:
#         graph_module.HyperGraph([input_node1, input_node2], output_node)
#     assert 'The network has a cycle.' in str(info.value)

# TODO: this test criterion may have some problem
def test_input_missing():
    input_node1 = Input()
    input_node2 = Input()
    output_node1 = MLPInteraction()(input_node1)
    output_node2 = MLPInteraction()(input_node2)
    output_node = ConcatenateInteraction()([output_node1, output_node2])
    output_node = RatingPredictionOptimizer()(output_node)

    with pytest.raises(ValueError) as info:
        graph_module.HyperGraph(input_node1, output_node)
    assert 'A required input is missing for HyperModel' in str(info.value)


def test_graph_basics():
    input_node = Input(shape=(30,))
    output_node = input_node
    output_node = MLPInteraction()(output_node)
    output_node = RatingPredictionOptimizer()(output_node)

    graph = graph_module.PlainGraph(input_node, output_node)
    model = graph.build_keras_graph().build(hp_module.HyperParameters())
    assert model.input_shape == (None, 30)
    assert model.output_shape == (None, )

