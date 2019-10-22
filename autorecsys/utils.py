from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import yaml
import tensorflow as tf

from autorecsys.searcher.core.hyperparameters import HyperParameters


def config_checker(config):
    # TODO: check configs
    return config


def load_config(raw_config):
    if isinstance(raw_config, dict):
        config = raw_config
    elif isinstance(raw_config, str):
        with open(os.path.join("./configs", raw_config + ".yaml"), "r", encoding='utf-8') as fr:
            config = yaml.load(fr)
    else:
        raise ValueError("Configuration should be a dict or a yaml filename!")

    return config_checker(config)


def convert_config(param_name_, param_info_):
    print(param_name_)
    print(param_info_)
    map_ = {'int': 'Int', 'float': 'Float', 'bool': 'Boolean', 'choice': 'Choice'}

    if 'default' not in param_info_:
        param_info_['default'] = param_info_['range'][0]

    if 'distribution' not in param_info_:
        param_info_["distribution"] = "choice"

    type_, distribution_, default_, range_ = param_info_['type'], param_info_["distribution"], \
                                             param_info_['default'], param_info_["range"]
    config = {'name': param_name_, 'default': default_}
    if distribution_ == 'discrete':
        config['values'] = range_
        config['ordered'] = True if type_ in {'int', 'float'} else False
    elif distribution_ == 'choice':
        type_ = 'choice'
        # config['values'] = range_
    else:
        config.update({'min_value': range_[0], 'max_value': range_[1], 'sampling': distribution_})
    type_ = map_[type_]
    print(config)
    print(type_)
    return config, type_


def extract_tunable_hps(config_dict):
    """
    Parse the autorecsys hyperparameters config dict to the Hyperparameters
    """
    hps = HyperParameters()
    for c_name, c_config_list in config_dict.items():  # component: mapper interactor optimizer
        for block in c_config_list:
            b_name, b_config = list(block.keys())[0], list(block.values())[0]
            if "params" in b_config.keys():
                for p_name, p_info in b_config["params"].items():
                    if isinstance(p_info, dict) and "range" in p_info:
                        p_config, p_type = convert_config('-'.join([c_name, b_name, p_name]), p_info)
                        with hps.name_scope(b_name):
                            method_to_call = getattr(hps, p_type)
                            method_to_call(**p_config)
    return hps


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not os.path.exists(path):
        os.mkdir(path)
    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        shutil.rmtree(path)
        os.mkdir(path)


def set_device(device_name):
    if device_name == "cpu":
        pass
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("Available GPUs: {}".format(gpus))
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        gpu_idx = int(device_name[-1])
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')