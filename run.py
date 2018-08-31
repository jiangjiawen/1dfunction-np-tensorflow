import tensorflow as tf
import numpy as np
from NP_arch import *
from NP_core import *

N = 5
x = np.array([-2, -1, 0, 1, 2])
y = np.sin(x)

sess = tf.Session()

x_context = tf.placeholder(tf.float32, [None, 1])
y_context = tf.placeholder(tf.float32, [None, 1])
x_target = tf.placeholder(tf.float32, [None, 1])
y_target = tf.placeholder(tf.float32, [None, 1])

train_op_and_loss = NP_init(
    x_context, y_context, x_target, y_target, learning_rate=0.001)

init = tf.global_variables_initializer()
sess.run(init)

n_iter = 5000
plot_freq = 200

n_draws = 50
# x_star = np.arange(-4, 4, 100, dtype=np.float32)
x_star = np.linspace(-4, 4, 100)
# eps_value = np.random.normal(n_draws * dim_r)
eps_value = np.random.randn(50, dim_r)
epsilon = tf.constant(eps_value, dtype=tf.float32)
predict_op = posterior_predict(x[:, np.newaxis], y[:, np.newaxis],
                               x_star[:, np.newaxis], epsilon)

df_pred_list = {}
for i in range(n_iter):
    import random
    N_list = [i + 1 for i in range(4)]
    N_context = random.sample(N_list, 1)[0]

    feed_dict = helper_context_and_target(x, y, N_context, x_context,
                                          y_context, x_target, y_target)

    a = sess.run(train_op_and_loss, feed_dict=feed_dict)

    # if i % 1e2 == 0:
    #     print(a[1])

    if i % plot_freq == 0:
        y_star_mat = sess.run(predict_op.mu)
        # print(y_star_mat)
        df_pred_list[i] = y_star_mat

# num = 0
# for k, v in df_pred_list.items():
#     for it in v:
#         num += len(it)
# print(num)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn

fig = plt.figure()
ax = plt.axes(xlim=(-5, 5), ylim=(-2, 2))
fig.set_tight_layout(True)

ax.scatter(x, y)

timetext = ax.text(2, 2, '')
line, = ax.plot([], [])
plt.ylabel('function values')
lines = []
for index in range(n_draws):
    lobj = ax.plot([], [], color='#826858')[0]
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([], [])
    return tuple(lines) + (timetext, )


def animate(i):
    label = 'timestep {0}'.format(i * plot_freq + plot_freq)
    timetext.set_text(label)
    data_y = df_pred_list[i * plot_freq]
    data_x = x_star
    for lnum, line in enumerate(lines):
        data_y_l = data_y[:, lnum]
        # plt.plot(data_x, data_y_j, color='#826858')
        line.set_data(data_x, data_y_l)
    return tuple(lines) + (timetext, )


nums_of_gif = n_iter / plot_freq
anim = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=np.arange(0, nums_of_gif),
    interval=400)
anim.save('result.gif', dpi=80, writer='imagemagick')
