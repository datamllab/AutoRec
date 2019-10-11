from autorecsys.mapper import *
from autorecsys.interaction import *
from autorecsys.optimizer import *

name_dict = {
    "LatentFactor": LatentFactorMapper,
    "MLP": MLPInteraction,
    "InnerProduct": InnerProductInteraction,
    "RatingPrediction": RatingPredictionOptimizer
}


def mapper_config( mapper_config_dict ):
    '''
    return: mapper dict{input_name:mapper}
    '''
    mapper_dict = { }
    for mapper_name in mapper_config_dict:
        print( mapper_name )
        tmp_config = mapper_config_dict[ mapper_name ]
        tmp_mapper = name_dict[ tmp_config[ "MapperType" ] ]( tmp_config[ "MapperParam" ][ "id_num" ],
                                                              tmp_config[ "MapperParam" ][ "LatentFactor_dim" ] )
        # print( tmp_mapper )
        # print( tmp_mapper( 1 ) )
        mapper_dict[ mapper_name ] = tmp_mapper
    return mapper_dict


def interaction_config( interaction_config_dict, mapper_dict ):
    '''
    return: mapper dict{input_name:mapper}
    '''
    interaction_dict = { }
    for interaction_name in interaction_config_dict:
        # print( interaction_name )
        tmp_type_input_config = {}
        tmp_config = interaction_config_dict[ interaction_name ]

        tmp_type_input_config[ "InteractionType" ] = name_dict[ tmp_config[ "InteractionType" ] ]
        tmp_type_input_config[ "Input" ] =  tmp_config[ "Input" ]
        tmp_type_input_config["InteractionParam"] = tmp_config[ "InteractionParam" ]
        # print( tmp_type_input_config  )

        interaction_dict[ interaction_name ] = tmp_type_input_config
    return interaction_dict


def recommender_config():
    pass
