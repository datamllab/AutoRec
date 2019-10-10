import tensorflow as tf


class baseMapper( tf.keras.layers.Layer ):
    def __init__(self):
        super(baseMapper, self).__init__()
        pass


    def call(self, x):
        pass




class LatentFactorMapper( baseMapper ):
    '''
    latent factor mapper for cateory datas
    '''
    def __init__(self, id_num, embedding_dim):
        super( LatentFactorMapper, self ).__init__()
        self.user_embedding = tf.keras.layers.Embedding( id_num, embedding_dim )

    def call(self, x):
        x = self.user_embedding( x )
        return x
