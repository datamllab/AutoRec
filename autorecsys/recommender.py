from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.pipeline.graph import HyperGraph

class RPRecommender(HyperGraph):
    """A Rating-Prediction HyperModel based on connected Blocks and HyperBlocks.
    # Arguments
        inputs: A list of input node(s) for the HyperGraph.
        outputs: A list of output node(s) for the HyperGraph.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CTRRecommender(HyperGraph):
    """A CTR-Prediction HyperModel based on connected Blocks and HyperBlocks.
    # Arguments
        inputs: A list of input node(s) for the HyperGraph.
        outputs: A list of output node(s) for the HyperGraph.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
