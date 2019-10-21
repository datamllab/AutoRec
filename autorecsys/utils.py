from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import yaml


def config_checker(config):
    # TODO: check configs
    return config


def load_config(raw_config):
    if isinstance(raw_config, dict):
        config = raw_config
    elif isinstance(raw_config, str):
        with open(os.path.join("./configs/", raw_config), "r", encoding='utf-8') as fr:
            config = yaml.load(fr)
    else:
        raise ValueError("Configuration should be a dict or a yaml filename!")

    return config_checker(config)


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not os.path.exists(path):
        os.mkdir(path)
    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        shutil.rmtree(path)
        os.mkdir(path)
