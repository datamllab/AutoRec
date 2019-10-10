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

    def call(self, vector):
        x = tf.reduce_sum( vector, axis = 1 )
        return x
