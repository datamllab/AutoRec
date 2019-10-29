from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.api import AutoSearch

if __name__ == "__main__":
    config_filename = "./examples/configs/mf_tune_config.yaml"
    searcher = AutoSearch(config_filename)
    searcher.search()