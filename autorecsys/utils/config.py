from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import yaml
import tensorflow as tf

from autorecsys.searcher.core.hyperparameters import HyperParameters


def env_config(config):
    load_config(config)


def config_checker(config):
    # TODO: check configs
    return config


def load_config(raw_config):
    if isinstance(raw_config, dict):
        config = raw_config
    elif isinstance(raw_config, str):
        with open(os.path.join("./examples/configs", raw_config + ".yaml"), "r", encoding='utf-8') as fr:
            config = yaml.load(fr)
    else:
        raise ValueError("Configuration should be a dict or a yaml filename!")

    return config_checker(config)


def convert_config(param_name_, param_info_):
    map_ = {'int': 'Int', 'float': 'Float', 'bool': 'Boolean', 'choice': 'Choice'}

    if 'default' not in param_info_:
        param_info_['default'] = param_info_['range'][0]

    if 'distribution' not in param_info_:
        param_info_["distribution"] = "choice"

    type_, distribution_, default_, range_ = param_info_['type'], param_info_["distribution"], \
                                             param_info_['default'], param_info_["range"]
    config = {'name': param_name_, 'default': default_}

    if distribution_ == 'choice':
        type_ = 'choice'
        config['values'] = range_
        config['ordered'] = True if type_ in {'int', 'float'} else False
    else:
        config.update({'min_value': range_[0], 'max_value': range_[1], 'sampling': distribution_})

    type_ = map_[type_]
    return config, type_


def extract_tunable_hps(config_dict):
    """
    Parse the autorecsys hyperparameters config dict to the Hyperparameters
    """
    hps = HyperParameters()
    for c_name, c_config_list in config_dict.items():  # component: mapper interactor optimizer
        for block_id, block in enumerate(c_config_list):
            b_name, b_config = list(block.keys())[0], list(block.values())[0]
            if b_name == "VirtualBlock" and len(b_config["block_choice"]) > 1:

                # add block choice
                p_config, p_type = convert_config('-'.join(
                    [c_name, str(block_id), b_name]
                ), {
                    "type": "int",
                    "range": list(range(len(b_config["block_choice"]))),
                    "distribution": "choice",
                    "default": 0
                }
                )
                method_to_call = getattr(hps, p_type)
                method_to_call(**p_config)

                for vb_id, vb in enumerate(b_config["block_choice"]):
                    vb_name, vb_config = list(vb.keys())[0], list(vb.values())[0]
                    if "params" in vb_config.keys():
                        for p_name, p_info in vb_config["params"].items():
                            if isinstance(p_info, dict) and "range" in p_info:
                                p_config, p_type = convert_config('-'.join(
                                    [c_name, str(block_id), b_name, str(vb_id), vb_name, p_name]
                                ), p_info)
                                method_to_call = getattr(hps, p_type)
                                method_to_call(**p_config)
            elif "params" in b_config.keys():
                for p_name, p_info in b_config["params"].items():
                    if isinstance(p_info, dict) and "range" in p_info:
                        p_config, p_type = convert_config('-'.join([c_name, str(block_id), b_name, p_name]), p_info)
                        method_to_call = getattr(hps, p_type)
                        method_to_call(**p_config)
    return hps


def set_tunable_hps(config_dict, hps):
    """
    Merge new hps to config dict
    """
    for c_name, c_config_list in config_dict.items():  # component: mapper interactor optimizer
        for block_id, block in enumerate(c_config_list):
            b_name, b_config = list(block.keys())[0], list(block.values())[0]

            if b_name == "VirtualBlock" and len(b_config["block_choice"]) > 1:

                # change virtual block to a specific block
                hp_name = '-'.join([c_name, str(block_id), b_name])
                vb_id = hps.values[hp_name]
                vb = b_config["block_choice"][vb_id]
                config_dict[c_name][block_id] = vb

                # set parameter for the selected block
                vb_name, vb_config = list(vb.keys())[0], list(vb.values())[0]
                if "params" in vb_config.keys():
                    for p_name, p_info in vb_config["params"].items():
                        if isinstance(p_info, dict) and "range" in p_info:
                            hp_name = '-'.join([c_name, str(block_id), b_name, str(vb_id), vb_name, p_name])
                            config_dict[c_name][block_id][vb_name]["params"][p_name] = [hps.values[hp_name]]

            elif "params" in b_config.keys():
                for p_name, p_info in b_config["params"].items():
                    if isinstance(p_info, dict) and "range" in p_info:
                        hp_name = '-'.join([c_name, str(block_id), b_name, p_name])
                        config_dict[c_name][block_id][b_name]["params"][p_name] = [hps.values[hp_name]]
    return config_dict


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
