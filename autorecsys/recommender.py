import tensorflow as tf
from autorecsys.mapper import *
from autorecsys.interaction import *
from autorecsys.optimizer import *




class recommender( tf.keras.Model ):
    def __init__( self, config ):
        super( recommender, self ).__init__()


        self.user_latentfactorMapper = LatentFactorMapper( userID_max, embedding_dim )
        self.item_latentfactorMapper = LatentFactorMapper( itemID_max, embedding_dim )
        self.interaction = InnerProductInteraction()
        self.optimizer = RatingPredictionOptimizer()

        # self.user_embedding = tf.keras.layers.Embedding( userID_max, embedding_dim )
        # self.item_embedding = tf.keras.layers.Embedding( itemID_max, embedding_dim )

    def call( self, userID, ItemID ):
        userID_embedding = self.user_latentfactorMapper( userID )
        itemID_embedding = self.item_latentfactorMapper( ItemID )
        # print( "embedding", userID_embedding, itemID_embedding )

        y_pred = self.interaction( userID_embedding, itemID_embedding )
        # print( "y_pred",y_pred )

        y_pred = self.optimizer( y_pred )
        # y_pred = tf.reduce_sum( y_pred, axis = 1 )
        # print( "y_pred_sum", y_pred )

        return y_pred



# class MF( tf.keras.Model ):
#     def __init__( self, userID_max, itemID_max, embedding_dim ):
#         super( MF, self ).__init__()
#         self.user_latentfactorMapper = LatentFactorMapper( userID_max, embedding_dim )
#         self.item_latentfactorMapper = LatentFactorMapper( itemID_max, embedding_dim )
#         self.interaction = InnerProductInteraction()
#         self.optimizer = RatingPredictionOptimizer()
#
#         # self.user_embedding = tf.keras.layers.Embedding( userID_max, embedding_dim )
#         # self.item_embedding = tf.keras.layers.Embedding( itemID_max, embedding_dim )
#
#     def call( self, userID, ItemID ):
#         userID_embedding = self.user_latentfactorMapper( userID )
#         itemID_embedding = self.item_latentfactorMapper( ItemID )
#         # print( "embedding", userID_embedding, itemID_embedding )
#
#         y_pred = self.interaction( userID_embedding, itemID_embedding )
#         # print( "y_pred",y_pred )
#
#         y_pred = self.optimizer( y_pred )
#         # y_pred = tf.reduce_sum( y_pred, axis = 1 )
#         # print( "y_pred_sum", y_pred )
#
#         return y_pred

