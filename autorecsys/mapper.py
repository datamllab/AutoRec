from abc import ABCMeta, abstractmethod
import tensorflow as tf



class BaseMapper( tf.keras.Model, metaclass=ABCMeta ):
# class BaseMapper( tf.keras.layers.Layer, metaclass=ABCMeta ):
    def __init__(self, **kwarg):
        super(BaseMapper, self).__init__()
        # self.name = name

    @abstractmethod
    def call(self, x):
        """call model."""



class LatentFactorMapper( BaseMapper ):
    '''
    latent factor mapper for cateory datas
    '''
    def __init__(self, id_num, embedding_dim):
        super( LatentFactorMapper, self ).__init__()
        self.user_embedding = tf.keras.layers.Embedding( id_num, embedding_dim )

    def call(self, x):
        x = self.user_embedding( x )
        return x


# class CompactLatentFactorMapper( BaseMapper ):
#     '''
#     latent factor mapper for cateory datas
#     '''
#     def __init__(self, id_num, embedding_dim):
#         super( LatentFactorMapper, self ).__init__()
#         self.user_embedding = tf.keras.layers.Embedding( id_num, embedding_dim )
#
#     def call(self, x):
#         x = self.user_embedding( x )
#         return x