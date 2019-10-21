from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def train(model, data):
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    avg_loss = []

    for step, (user_id, item_id, y) in enumerate(data.take(10)):
        feat_dict = {"user_id": user_id, "item_id": item_id}
        with tf.GradientTape() as tape:
            y_pred = model(feat_dict)
            loss = tf.keras.losses.MSE(y_pred["rating"], y)
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        avg_loss.append(float(loss))
        print("Step: {}, avg_loss: {}, loss: {}".format(
            step,
            sum(avg_loss[-1000:]) / min(1000., step + 1),
            loss
            )
        )
    return model
