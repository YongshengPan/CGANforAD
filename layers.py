import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def instance_norm1D(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
        return out

def instance_norm2D(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
        return out

def instance_norm3D(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
        return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d",
                   do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm2D(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
        
        if do_norm:
            conv = instance_norm2D(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_conv3d(inputconv, o_d=64, f_h=7, f_w=7, f_d=7, s_h=1, s_w=1, s_d=1, stddev=0.5, padding="VALID", name="conv3d",
                   do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.layers.conv3d(inputconv, o_d, [f_h, f_w, f_d], [s_h, s_w, s_d], padding, activation=None,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm3D(conv)
            # conv = tf.layers.batch_normalization(conv, epsilon=1e-5, training=True, name="batch_norm")

        if do_relu:
            if relufactor == 0:
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv


def general_deconv3d(inputconv, outshape, o_d=64, f_h=7, f_w=7, f_d=7, s_h=1, s_w=1, s_d=1, stddev=0.02, padding="VALID",
                     name="deconv3d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.layers.conv3d_transpose(inputconv, o_d, [f_h, f_w, f_d], [s_h, s_w, s_d], padding, activation=None,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  bias_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = instance_norm3D(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            # conv = tf.layers.batch_normalization(conv, epsilon=1e-5, trainable=True, scope="batch_norm")
        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def fc_op(x, name, n_out, activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        fc = tf.matmul(x, w) + b
        out = activation(fc)

    return fc, out
