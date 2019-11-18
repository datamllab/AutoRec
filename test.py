import tensorflow as tf



class MyModel(tf.keras.Model):
    def call(self, inputs):
        return [inputs['a'], 2 * inputs['b']]


lr = 0.001


inputs = {'a': tf.range(5.0), 'b': tf.range(5.0, 10.0)}
labels = [tf.range(5, 10), tf.range(5, 10)]


model = MyModel()
model.compile(tf.optimizers.Adam(learning_rate=lr), 'mean_squared_error')
model.fit(x=inputs, y=labels)
model.summary()

