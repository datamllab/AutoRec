import logging

LOGGER = logging.getLogger(__name__)
TYPE_MAP = {'int': int, 'float': float, 'str': str, 'list': list, 'tuple': tuple, 'bool': bool}
CANT_BE_SET = -1


def check_valid_params(name, x, param_info, skip_range_detect):
    param_type = TYPE_MAP[param_info['type']]
    try:
        x = param_type(x)
    except ValueError as e:
        LOGGER.exception(f'can not cast {name} to {param_type}')
        raise e
    param_range = param_info.get('range', None)
    if param_range == CANT_BE_SET:
        raise TypeError(f'{name} can not be set from config files')
    if not skip_range_detect:
        if isinstance(param_range, tuple):
            if x not in param_range:
                raise ValueError(f'{name} must be in {param_range}, {x} doesn\'t')
        elif isinstance(param_range, list):
            low, high = param_range
            if x < low or x > high:
                raise ValueError(f'{name} valid range: x>={low} && x<={high}')
        else:
            raise NotImplementedError(f'code error: the param\'range of a model must be tuple, list')
    return x
