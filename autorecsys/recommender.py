import tensorflow as tf
from autorecsys.mapper import *
from autorecsys.interaction import *
from autorecsys.optimizer import *


class recommender(tf.keras.Model):
    def __init__(self, mapper_list, interaction_list, optimizer_list):
        super(recommender, self).__init__()
        self.mapper_list = mapper_list
        self.interaction_list = {}
        self.optimizer_list = {}

        # for mapper in mapper_list:
        #     self.mapper_list[ mapper ] = mapper_list[ "mapper" ]

        # self.user_latentfactorMapper = LatentFactorMapper( userID_max, embedding_dim )
        # self.item_latentfactorMapper = LatentFactorMapper( itemID_max, embedding_dim )

        self.user_latentfactorMapper = self.mapper_list["user_id"]
        self.item_latentfactorMapper = self.mapper_list["item_id"]

        # self.interaction = InnerProductInteraction()
        self.interaction = MLPInteraction()

        self.optimizer = RatingPredictionOptimizer()

        # self.user_embedding = tf.keras.layers.Embedding( userID_max, embedding_dim )
        # self.item_embedding = tf.keras.layers.Embedding( itemID_max, embedding_dim )

    def call(self, userID, ItemID):
        userID_embedding = self.user_latentfactorMapper(userID)
        itemID_embedding = self.item_latentfactorMapper(ItemID)
        # print( "embedding", userID_embedding, itemID_embedding )

        y_pred = self.interaction(userID_embedding, itemID_embedding)
        # print( "y_pred",y_pred )

        y_pred = self.optimizer(y_pred)
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
