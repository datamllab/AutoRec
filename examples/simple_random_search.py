from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.api import AutoSearch

if __name__ == "__main__":
    config_filename = "./examples/configs/random_search_config.yaml"
    search_api = AutoSearch(config_filename)
    results = search_api.search()
    print(results)
