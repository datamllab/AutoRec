from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU
os.chdir("../../examples")

logger = logging.getLogger(__name__)
# tf.random.set_seed(1)


class CTRTestModels(unittest.TestCase):

    def setUp(self):
        super(CTRTestModels, self).setUp()
        self.ctr_model = {'autoint': 'ctr_autoint.py',
                          'autorec': 'ctr_autorec.py',
                          'crossnet': 'ctr_crossnet.py',
                          'deepfm': 'ctr_deepfm.py',
                          'dlrm': 'ctr_dlrm.py',
                          'neumf': 'ctr_neumf.py'}

    def test_ctr_autoint(self):
        """
        Test class in ctr_autoint.py
        """
        try:
            exec(open(self.ctr_model['autoint']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_ctr_autorec(self):
        """
        Test class in ctr_autorec.py
        """
        try:
            exec(open(self.ctr_model['autorec']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_ctr_crossnet(self):
        """
        Test class in ctr_crossnet.py
        """
        try:
            exec(open(self.ctr_model['crossnet']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_ctr_deepfm(self):
        """
        Test class in ctr_deepfm.py
        """
        try:
            exec(open(self.ctr_model['deepfm']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_ctr_dlrm(self):
        """
        Test class in ctr_dlrm.py
        """
        try:
            exec(open(self.ctr_model['dlrm']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_ctr_neumf(self):
        """
        Test class in ctr_neumf.py
        """
        try:
            exec(open(self.ctr_model['neumf']).read())
        except RuntimeError:
            assert False, 'Runtime Error'


class RPTestModels(unittest.TestCase):

    def setUp(self):
        super(RPTestModels, self).setUp()
        self.rp_model = {'autorec': 'rp_autorec.py',
                         'mf': 'rp_autorec.py',
                         'neumf': 'rp_neumf.py'}

    def test_rp_autorec(self):
        """
        Test class in rp_autorec.py
        """
        try:
            exec(open(self.rp_model['autorec']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_rp_mf(self):
        """
        Test class in rp_mf.py
        """
        try:
            exec(open(self.rp_model['mf']).read())
        except RuntimeError:
            assert False, 'Runtime Error'

    def test_rp_neumf(self):
        """
        Test class in rp_neumf.py
        """
        try:
            exec(open(self.rp_model['neumf']).read())
        except RuntimeError:
            assert False, 'Runtime Error'
