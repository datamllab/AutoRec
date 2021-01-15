from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.pipeline.graph import HyperGraph


class RPRecommender(HyperGraph):  # pragma: no cover
    """A rating prediction HyperModel based on connected Blocks and HyperBlocks.

    # Arguments
        inputs (list): A list of input node(s) for the HyperGraph.
        outputs (list): A list of output node(s) for the HyperGraph.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CTRRecommender(HyperGraph):  # pragma: no cover
    """A CTR (click-through rate) prediction HyperModel based on connected Blocks and HyperBlocks.

    # Arguments
        inputs (list): A list of input node(s) for the HyperGraph.
        outputs (list): A list of output node(s) for the HyperGraph.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
