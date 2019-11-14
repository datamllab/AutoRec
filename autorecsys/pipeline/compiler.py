from autokaggle.hypermodel import node as node_module
from autokaggle.hypermodel import block as block_module


def preprocess_input(preprocess_block):
    if not isinstance(preprocess_block.inputs[0], node_module.Input):
        raise TypeError('Preprocess block can only be used with Input.')

    preprocess_block.md = preprocess_block.inputs[0].md


def feature_engineering_input(fe_block):
    # from the functional API design, one node could have several output blocks, but it always
    # have one exact input block
    # if len(fe_block.inputs) != 1 or len(fe_block.inputs[0].in_blocks) != 1:
    if len(fe_block.inputs) != 1:
        raise ValueError('FeatureEngineering block can only have one input, and it should be Preprocess block')
    in_block = fe_block.inputs[0].in_blocks[0]
    if not isinstance(in_block, block_module.Preprocess):
        raise TypeError(f'FeatureEngineering block can only have one input, and it should be Preprocess block,'
                        f'but got input block: {in_block.__class__}')
    fe_block.md = in_block.md


def lgb_input(lgb_block):
    # from the functional API design, one node could have several output blocks, but it always
    # have one exact input block
    # fetch the md from its inputs
    if len(lgb_block.inputs) != 1:
        raise ValueError('LightGBMBlock can only have one input node')
    in_block = lgb_block.inputs[0].in_blocks[0] \
        if not isinstance(lgb_block.inputs[0], node_module.Input) else lgb_block.inputs[0]
    lgb_block.md = in_block.md


# Compile the Graph. Fetch the MetaData from the input.
FETCH_MD = {**{block_module.Preprocess: preprocess_input, block_module.FeatureEngineering: feature_engineering_input,
               block_module.LightGBMBlock: lgb_input}}
