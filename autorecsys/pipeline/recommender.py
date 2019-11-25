from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.utils import load_config
from autorecsys.pipeline.graph import HyperGraph

import logging
# from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CFRecommender(HyperGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CTRRecommender(HyperGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
