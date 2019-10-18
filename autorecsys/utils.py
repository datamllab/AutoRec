import os
import yaml


def config_checker(config):
    # TODO: check config
    return config


def load_config(config_filename):
    with open(os.path.join("./config/", config_filename), "r", encoding='utf-8') as fr:
        config = yaml.load(fr)
    return config_checker(config)