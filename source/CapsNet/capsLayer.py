import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim


class CapsLayer(object):

    def __init__(self, layer_type='PrimaryCaps'):
        self.layer_type = layer_type
    
    def __call__(self, input, num_outputs=32, vec_len=8, kernel_size=9, stride=2,):
        if self.layer_type == 'PrimaryCaps':
            # input = [batch_size, 20, 20, 256]
            batch_size = input.shape[0].value
            capsules = slim.conv2d(input, num_outputs * vec_len, kernel_size, stride, padding='VALID')
            # capsules = [batch_size, 6, 6, 32*8]

            capsules = tf.reshape(capsules, (batch_size, -1, vec_len, 1))
            # capsules = [batchsize, 6 * 6 * 32, 8, 1]

            return squash(capsules)

        elif self.layer_type == 'DigitCaps':
            # input = [batch_size, 6 * 6 * 32, 8, 1]
            batch_size = input.shape[0].value
            input_caps_num = input.shape[1].value
            input_vec_len = input.shape[2].value


            with tf.variable_scope('routing'):
                # b_IJ = tf.constant(np.zeros([batch_size, input_caps_num, num_outputs, 1, 1], dtype=np.float32))
                # capsules = routing(tf.reshape(input, shape=(batch_size, -1, 1, input_vec_len, 1)), b_IJ)
                # capsules = [batch_size, num_outputs, vec_len, 1]
                capsules = routing(input, 10, 16)

            return tf.squeeze(capsules, axis=1)

def routing(input, out_caps_num, out_vec_len):
    """ The routing algorithm.
    Args:
        input: capsules with [batch_size, caps_num, vec_len, 1]
        out_caps_num: the number of output capsules
        out_vec_len: the vector length of output capsules (defaule vec_len)
    output:
        out_capsules: capsules with [batch_size, out_caps_num, out_vec_len]
    """
    iter_routing = 3
    in_shape = input.get_shape()
    # in_shape = [batch_size, caps_num, vec_len, 1]

    W = tf.get_variable('Weight', shape=(1, in_shape[1], out_caps_num*out_vec_len, in_shape[2]))
    # W = [1, caps_num, out_caps_num * out_vec_len, vec_len, 1]

    biases = tf.get_variable('bias', shape=(1, 1, out_caps_num, out_vec_len))
    # biases = [1, 1, out_caps_num, out_vec_len]

    W = tf.tile(W, [in_shape[0], 1, 1, 1])

    # u_hat = Wij * u
    u_hat = tf.matmul(W, input)
    # u_hat = [batch_size,  caps_num, out_caps_num * out_vec_len, 1]

    u_hat = tf.reshape(u_hat, shape=(in_shape[0], in_shape[1], out_caps_num, out_vec_len))
    # u_hat = [batch_size, caps_num, out_caps_num, out_vec_len]

    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    b_IJ = tf.constant(np.zeros([in_shape[0], in_shape[1], out_caps_num, 1], dtype=np.float32))
    # b_IJ = [batch_size, caps_num, out_caps_num, 1]
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):

            if r_iter == iter_routing - 1:
                c_IJ = tf.nn.softmax(b_IJ, axis=2)
                # c_IJ = [batch_size, caps_num, out_caps_num, 1]

                s_j = tf.multiply(c_IJ, u_hat)
                # s_j = [batch_size, caps_num, out_caps_num, out_vec_len]

                s_j = tf.reduce_sum(s_j, axis=1, keepdims=True) + biases
                # not clear, paper does not mention biases
                # s_j = [batch_size, 1, out_caps_num, out_vec_len]

                v_j = squash(s_j)
                # v_j = [batch_size, 1, out_caps_num, out_vec_len]
            else:
                c_IJ = tf.nn.softmax(b_IJ, axis=2)
                # c_IJ = [batch_size, caps_num, out_caps_num, 1]

                s_j = tf.multiply(c_IJ, u_hat_stopped)
                # s_j = [batch_size, caps_num, out_caps_num, out_vec_len]

                s_j = tf.reduce_sum(s_j, axis=1, keepdims=True) + biases
                # not clear, paper does not mention biases
                # s_j = [batch_size, 1, out_caps_num, out_vec_len]

                v_j = squash(s_j)
                # v_j = [batch_size, 1, out_caps_num, out_vec_len]

                # bij = bij + u_hat * vj
                v_J_tiled = tf.tile(v_j, [1, in_shape[1], 1, 1])
                # v_j_tiled = [batch_size, caps_num, out_caps_num, out_vec_len]

                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # u_produce_v = [batch_size, caps_num, out_caps_num, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return v_j

def routing_old(input, b_IJ):

    iter_routing = 3
    batch_size = input.get_shape()[0]

    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, 160, 1, 1])
    assert input.get_shape() == [batch_size, 1152, 160, 8, 1]

    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])
    assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                assert s_J.get_shape() == [batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [batch_size, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                assert u_produce_v.get_shape() == [batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    # vector = [batchsize, 6 * 6 * 32, 8, 1]

    # vj = (||sj||^2 / (1 + ||sj||^2)) * (sj / ||sj||)

    epsilon = 1e-9

    vec_squared_norm = tf.reduce_sum(tf.square(vector), axis=-2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    
    return vec_squashed
