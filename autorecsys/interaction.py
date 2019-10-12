import tensorflow as tf
from abc import ABCMeta, abstractmethod


class BaseInteraction( tf.keras.Model, metaclass=ABCMeta ):
    def __init__(self):
        super(BaseInteraction, self).__init__()
        pass

    @abstractmethod
    def call(self, x):
        """call model."""



class InnerProductInteraction( BaseInteraction ):
    '''
    latent factor mapper for cateory datas
    '''
    def __init__(self):
        super( InnerProductInteraction, self ).__init__()

    def call(self, embed1, embed2):
        x = embed1 * embed2
        return x


class MLPInteraction_legency( BaseInteraction ):
    '''
    latent factor mapper for cateory datas
    '''
    def __init__(self):
        super( MLPInteraction, self ).__init__()
        self.dense_layers = []
        self.dense_layers.append( tf.keras.layers.Dense(128) )
        self.dense_layers.append( tf.keras.layers.Dense( 128 ) )
        self.dense_layers.append( tf.keras.layers.Dense( 128 ) )
        self.dense_layers.append( tf.keras.layers.Dense( 64 ) )

    def call(self, embeds ):
        x = tf.concat( embeds, axis = 1 )
        for layer in self.dense_layers:
            x = layer( x )
        return x



class MLPInteraction( BaseInteraction ):
    '''
    latent factor mapper for cateory datas
    '''
    def __init__(self):
        super( MLPInteraction, self ).__init__(  )
        self.dense_layers = []
        self.dense_layers.append( tf.keras.layers.Dense(128) )
        self.dense_layers.append( tf.keras.layers.Dense( 128 ) )
        self.dense_layers.append( tf.keras.layers.Dense( 128 ) )
        self.dense_layers.append( tf.keras.layers.Dense( 64 ) )


    def call(self, embeds):
        x = tf.concat( embeds , axis = 1 )
        for layer in self.dense_layers:
            x = layer( x )
        return x
