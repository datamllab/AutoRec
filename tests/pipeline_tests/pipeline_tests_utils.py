import tensorflow as tf
import numpy as np


def layer_test(interactor, input_data, ans, p, name=None):
    if input_data is None:
        raise ValueError('input data is None')

    # expected output dtype
    assert tf.is_tensor(ans)

    # expected output shape
    if input_data.ndim == 3:
        [a, b] = input_data.shape[0], input_data.shape[1] * input_data.shape[2]
        if name == 'MLP':

            # Dense shape:
            assert ans.shape[0] == a
            assert ans.shape[1] == p[0]

            # HP values
            assert tf.equal(p[0], interactor.get_state()['units'])
            assert tf.equal(p[1], interactor.get_state()['num_layers'])
            assert tf.equal(p[2], interactor.get_state()['use_batchnorm'])
            assert tf.equal(p[3], interactor.get_state()['dropout_rate'])

            # Relu: all positive entries
            assert tf.reduce_all(tf.greater_equal(ans, 0))

            # glorot weight: https://keras.rstudio.com/reference/initializer_glorot_uniform.html
            weights = np.sqrt(6 / (input_data.shape[2] + p[0]))

            # Dense weights: (input sum entries * dense layer dim * weights range)
            for _ in range(p[1]):
                x = tf.reduce_sum(input_data) * p[0]
                assert tf.greater_equal(tf.reduce_sum(ans), x * -weights)
                assert tf.less_equal(tf.reduce_sum(ans), x * weights)

        elif name == 'FM':
            # shape tf.reduce_sum
            assert ans.shape[0] == a
            assert ans.shape[1] == 1

            # HP values
            assert tf.equal(p[0], interactor.get_state()['embedding_dim'])

            # sum of square
            assert tf.reduce_all(tf.greater_equal(ans, 0))

            # Dense weights / assume "glorot_uniform"
            dw = tf.ones([input_data.shape[2], p[0]], dtype=tf.float32)

            # glorot weight: https://keras.rstudio.com/reference/initializer_glorot_uniform.html
            weights = np.sqrt(6 / (input_data.shape[2] + p[0]))

            for i in range(a):  # for every batch
                x = tf.tensordot(input_data[i], dw * weights, axes=1)
                square_of_sum = tf.square(tf.reduce_sum(x))
                sum_of_square = tf.reduce_sum(x * x)
                cross_term = square_of_sum - sum_of_square
                output = 0.5 * tf.reduce_sum(cross_term, keepdims=False)

                # tf.square makes output always positive
                assert tf.less_equal(ans[i], output)

        elif name == 'CrossNet':

            # output shape
            assert ans.shape[0] == a
            assert ans.shape[1] == b

            # Dense weights / assume "glorot_uniform"
            dw = tf.ones([b, 1], dtype=tf.float32)
            inputs = tf.reshape(input_data, [-1, b])

            # glorot weight: https://keras.rstudio.com/reference/initializer_glorot_uniform.html
            weights = np.sqrt(6 / (input_data.shape[2] + p[0]))

            for _ in range(p[0]):  # for every hp layer
                for i in range(a):  # for every batch
                    p_pre_output_emb = tf.tensordot(inputs[i], dw * weights, axes=1)
                    p_cross_dot = tf.math.multiply(inputs[i], p_pre_output_emb)
                    p_output = p_cross_dot + inputs[i]

                    n_pre_output_emb = tf.tensordot(inputs[i], dw * -weights, axes=1)
                    n_cross_dot = tf.math.multiply(inputs[i], n_pre_output_emb)
                    n_output = n_cross_dot + inputs[i]

                    # skip bias term -> initialized to zeroes

                    # tf.square makes output always positive
                    assert tf.reduce_all(tf.less_equal(ans[i], p_output))
                    assert tf.reduce_all(tf.greater_equal(ans[i], n_output))

            # HP values
            assert tf.equal(p[0], interactor.get_state()['layer_num'])

    else:
        raise ValueError('input data shape is not 3d')

    return
