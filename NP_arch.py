import tensorflow as tf
import collections


def h(inputs, dim_h_hidden, dim_r):
    Dense1 = tf.layers.dense(
        inputs,
        dim_h_hidden,
        activation=tf.nn.sigmoid,
        name='encoder_layer1',
        reuse=tf.AUTO_REUSE)
    Dense2 = tf.layers.dense(
        Dense1, dim_r, name="encoder_layer2", reuse=tf.AUTO_REUSE)
    return Dense2


def aggregate_r(inputs_h):
    aggr_r = tf.reduce_mean(inputs_h, axis=0)
    aggr_r = tf.reshape(aggr_r, [1, -1])
    return aggr_r


def get_z_params(inputs_r, dim_z):
    mu = tf.layers.dense(
        inputs_r, dim_z, name="z_params_mu", reuse=tf.AUTO_REUSE)
    sigma = tf.layers.dense(
        inputs_r, dim_z, name="z_parms_sigma", reuse=tf.AUTO_REUSE)
    sigma = tf.nn.softplus(sigma)
    Point = collections.namedtuple('Point', ['mu', 'sigma'])
    p = Point(mu, sigma)
    return p


def g(z_sample, x_star, dim_g_hidden, noise_sd=0.05):
    n_draws = z_sample.get_shape().as_list()[0]
    N_star = tf.shape(x_star)[0]

    z_sample_rep = tf.expand_dims(z_sample, axis=1)
    z_sample_rep = tf.tile(z_sample_rep, [1, N_star, 1])

    x_star_rep = tf.expand_dims(x_star, axis=0)
    x_star_rep = tf.tile(x_star_rep, [n_draws, 1, 1])

    input_g = tf.concat([x_star_rep, z_sample_rep], 2)
    hidden_g = tf.layers.dense(
        input_g,
        dim_g_hidden,
        activation=tf.nn.sigmoid,
        name="decoder_layer1",
        reuse=tf.AUTO_REUSE)

    mu_star = tf.layers.dense(
        hidden_g, 1, name="decoder_layer2", reuse=tf.AUTO_REUSE)
    mu_star = tf.squeeze(mu_star, axis=2)
    mu_star = tf.transpose(mu_star)

    sigma_star = tf.constant(noise_sd, dtype=tf.float32)

    Point = collections.namedtuple('Point', ['mu', 'sigma'])
    p = Point(mu_star, sigma_star)

    return p
