## code=utf-8
from queue import Queue
import json
import yaml

import tensorflow as tf
from autorecsys.mapper import *
from autorecsys.interaction import *
from autorecsys.optimizer import *


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print( physical_devices )
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


used_column = [ 0, 1, 2 ]
record_defaults = [ tf.int32, tf.int32, tf.float32 ]

# data = tf.data.experimental.CsvDataset( "./datasets/ml-1m/ratings.dat",  record_defaults, header = False, field_delim = "::", select_cols = used_column  )
data = tf.data.experimental.CsvDataset( "./datasets/ml-1m/ratings.dat", record_defaults, field_delim=",",
                                        select_cols=used_column )
data = data.repeat().shuffle( buffer_size=1000 ).batch( batch_size = 10240 ).prefetch( buffer_size=5 )

# print( 'start' )
# for step, item in enumerate( data.take( 5 ) ):
#     print( item )

# with  open('./config.json','r',encoding='utf-8') as fr:
#     config = json.load( fr )
#     print( config )
#
#
# with open( "./config.yaml", "w", encoding='utf-8' ) as fw:
#     yaml.dump( config, fw )


with open( "./config.yaml", "r", encoding='utf-8' ) as fr:
   config =  yaml.load( fr )
   print( config )
   print( config[ "Mapper" ] )
   print( config[ "Interaction" ] )
   print( config[ "Optimizer" ] )

class MF( tf.keras.Model ):
    def __init__( self, userID_max, itemID_max, embedding_dim ):
        super( MF, self ).__init__()

        self.user_latentfactorMapper = LatentFactorMapper( userID_max, embedding_dim )
        self.item_latentfactorMapper = LatentFactorMapper( itemID_max, embedding_dim )

        # self.interaction = InnerProductInteraction()
        self.interaction = MLPInteraction()


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





# Cross-Entropy Loss.
model = MF(100000, 100000, 10)
optimizer = tf.optimizers.Adam(learning_rate=0.01)


avg_loss =  []

for step, (user_id, item_id, y)  in enumerate( data.take( 100000 ) ):
    # print( user_id, item_id, y )
    with tf.GradientTape() as tape:
        y_pred = model( user_id, item_id )
        # print(  y_pred )
        loss = tf.keras.losses.MSE( y_pred, y )
    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # print( step, 'loss:', float( loss ) )
    avg_loss.append( float(loss) )
    print(step, "avg_loss", sum( avg_loss[-1000:] ) / min( 1000., step + 1 ), 'loss:', float(loss))
    # print( step,  'loss:', float( loss ) )
# Stochastic gradient descent optimizer.



# print( tf.__version__ )
