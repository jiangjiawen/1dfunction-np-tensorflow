import tensorflow as tf
import numpy as np
import random
import collections
from NP_arch import h, aggregate_r, get_z_params, g

dim_r = 2
dim_z = 2
dim_h_hidden = 8
dim_g_hidden = 8


def map_xy_to_z_params(x, y):
    z_inputs = tf.concat([x, y], 1)
    m_h = h(inputs=z_inputs, dim_h_hidden=dim_h_hidden, dim_r=dim_r)
    m_aggr = aggregate_r(inputs_h=m_h)
    m_get_z = get_z_params(inputs_r=m_aggr, dim_z=dim_z)
    return m_get_z


def NP_init(x_context, y_context, x_target, y_target, learning_rate=0.001):
    x_all = tf.concat([x_context, x_target], 0)
    y_all = tf.concat([y_context, y_target], 0)

    z_context = map_xy_to_z_params(x_context, y_context)
    z_all = map_xy_to_z_params(x_all, y_all)

    epsilon = tf.random_normal([7, dim_z])
    z_sample = tf.multiply(z_all.sigma, epsilon)
    z_sample = tf.add(z_sample, z_all.mu)

    y_pred_params = g(z_sample, x_target, dim_g_hidden=dim_g_hidden)
    loglik = loglikelihood(y_target, y_pred_params)
    KL_loss = KLqp_gaussian(z_all.mu, z_all.sigma, z_context.mu,
                            z_context.sigma)
    loss = tf.negative(loglik) + KL_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    Point = collections.namedtuple('Point', ['train_op', 'loss'])
    p = Point(train_op, loss)

    return p


def KLqp_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    sigma2_q = tf.square(sigma_q) + 1e-16
    sigma2_p = tf.square(sigma_p) + 1e-16
    temp = sigma2_q / sigma2_p + tf.square(
        mu_q - mu_p) / sigma2_p - 1.0 + tf.log(sigma2_p / sigma2_q + 1e-16)
    temp = 0.5 * tf.reduce_sum(temp)
    return temp


def loglikelihood(y_star, y_pred_params):
    p_normal = tf.distributions.Normal(
        loc=y_pred_params.mu, scale=y_pred_params.sigma)
    loglik = p_normal.log_prob(y_star)
    loglik = tf.reduce_sum(loglik, axis=0)
    loglik = tf.reduce_mean(loglik)
    return loglik


def prior_predict(x_star_value, epsilon=None, n_draws=1):
    N_star = len(x_star_value)
    x_star = tf.constant(x_star_value, dtype=tf.float32)

    if epsilon is None:
        epsilon = tf.random_normal([n_draws, dim_z])

    z_sample = epsilon
    y_star = g(z_sample, x_star, dim_g_hidden=dim_g_hidden)
    return y_star


def posterior_predict(x, y, x_star_value, epsilon=None, n_draws=1):
    x_obs = tf.constant(x, dtype=tf.float32)
    y_obs = tf.constant(y, dtype=tf.float32)
    x_star = tf.constant(x_star_value, dtype=tf.float32)

    z_params = map_xy_to_z_params(x_obs, y_obs)

    if epsilon is None:
        epsilon = tf.random_normal([n_draws, dim_z])

    z_sample = tf.multiply(z_params.sigma, epsilon)
    z_sample = tf.add(z_params.mu, z_sample)

    y_star = g(z_sample, x_star, dim_g_hidden=dim_g_hidden)
    return y_star


def helper_context_and_target(x, y, N_context, x_context, y_context, x_target,
                              y_target):
    N = len(y)
    listN = [i for i in range(N)]
    context_set = random.sample(listN, N_context)
    target_set = list(set(listN) - set(context_set))
    return {
        x_context: x[context_set][:, np.newaxis],
        y_context: y[context_set][:, np.newaxis],
        x_target: x[target_set][:, np.newaxis],
        y_target: y[target_set][:, np.newaxis]
    }
