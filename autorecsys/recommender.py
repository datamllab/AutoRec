from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.pipeline.graph import HyperGraph

class RPRecommender(HyperGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CTRRecommender(HyperGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
