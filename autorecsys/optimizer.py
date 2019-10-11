import tensorflow as tf


class baseOptimizer( tf.keras.layers.Layer ):
    def __init__(self):
        super(baseOptimizer, self).__init__()
        pass


    def call(self, x):
        pass




class RatingPredictionOptimizer( baseOptimizer ):
    '''
    latent factor mapper for cateory datas
    '''
    def __init__(self):
        super( RatingPredictionOptimizer, self ).__init__()
        # self.user_embed = user_embed
        # self.item_embed = item_embed
        self.dense_layer = tf.keras.layers.Dense( 1 )

    def call(self, embeds):
        x = tf.concat( embeds, axis=1 )
        x = self.dense_layer( x )
        x = tf.reshape( x, [-1] )
        # x = tf.reduce_sum( x, axis = 1 )
        return x
