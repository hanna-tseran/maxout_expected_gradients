from datetime import datetime
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import random
import sys
import tensorflow as tf
import tensorflow_addons as tfa

random.seed(7726)
np.random.seed(7726)

COLOR_LIST = ['xkcd:light purple', 'xkcd:pale brown', 'xkcd:orange', 'xkcd:blue', 'xkcd:irish green', 'xkcd:red',
    'xkcd:yellow orange', 'xkcd:purple', 'xkcd:sky blue', 'xkcd:red brown', 'xkcd:magenta',
    'xkcd:green/yellow', 'xkcd:pink', 'xkcd:toxic green', 'xkcd:marine']

comm = MPI.COMM_WORLD

WIDTH = 2
NUMBER_OF_RUNS = 7
NUMBER_OF_X_RUNS = 2
K = 5
NETWORK_TYPE = 'fc' #'fc' for a fully-connected network or 'cnn' for a convolutional neural network

# Reciprocals of the constants from the Table 4 in paper.
# m_dict corresponds to the suggested optimal value.
s_dict = {2: 0.61, 3: 0.48, 4: 0.4, 5: 0.36}
l_dict = {2: 2.75, 3: 5.19, 4: 8.28, 5: 12}
m_dict = {2: 1., 3: 0.78391, 4:0.64461, 5: 0.55555}

c_to_color = {
    s_dict[K]:'xkcd:blue',
    m_dict[K]: 'xkcd:red',
    1: 'xkcd:irish green',
    2: 'xkcd:yellow orange',
    l_dict[K]: 'xkcd:sky blue',
    0.1: 'xkcd:orange',
    100: 'xkcd:purple'
}
c_array = list(set(c_to_color.keys()))
c_array.sort()

depth_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]

if NETWORK_TYPE == 'fc':
    INPUT_SHAPE = [WIDTH]
elif NETWORK_TYPE == 'cnn':
    INPUT_SHAPE = (32, 32, 3)
else:
    raise Exception('Wrong NETWORK_TYPE')
OUTPUT_SHAPE = 1
KERNEL_SIZE = 3

weights_initializer = lambda c, fan_in: tf.keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(c / fan_in))
cnn_weights_initializer = lambda c, k, fan_in: tf.keras.initializers.RandomNormal(
    mean=0., stddev=np.sqrt(c / ((k**2) * fan_in)))

def get_net(depth, c):
    if NETWORK_TYPE == 'fc':
        #########################################################################
        fan_in = INPUT_SHAPE[-1]
        layers = []
        for _ in range(depth):
            # This combination == one maxout layer
            layers.append(tf.keras.layers.Dense(units=WIDTH * K, activation='linear',
                kernel_initializer=weights_initializer(c, fan_in),
                bias_initializer=tf.zeros_initializer()))
            layers.append(tfa.layers.Maxout(WIDTH))

            fan_in = WIDTH

        # The last linear layer
        layers.append(tf.keras.layers.Dense(units=OUTPUT_SHAPE,
            kernel_initializer=weights_initializer(1., fan_in),
            bias_initializer=tf.zeros_initializer()))

        return tf.keras.models.Sequential(layers)
        #########################################################################

    elif NETWORK_TYPE == 'cnn':
        #########################################################################
        fan_in = INPUT_SHAPE[-1]
        filter_size = INPUT_SHAPE[0]
        layers = []
        channels = WIDTH
        kernel_size = KERNEL_SIZE
        for _ in range(depth):
            # This combination == one maxout layer
            layers.append(tf.keras.layers.Conv2D(
                filters=channels * K,
                kernel_size=kernel_size,
                padding='same',
                activation='linear',
                kernel_initializer=cnn_weights_initializer(c=c, k=kernel_size, fan_in=fan_in),
                bias_initializer=tf.zeros_initializer()
                ))
            layers.append(tfa.layers.Maxout(channels, axis=-1))

            fan_in = channels

        fan_in = channels * filter_size**2
        layers.append(tf.keras.layers.Flatten())

        # The last linear layer
        layers.append(tf.keras.layers.Dense(units=OUTPUT_SHAPE,
            kernel_initializer=weights_initializer(1., fan_in),
            bias_initializer=tf.zeros_initializer()))

        return tf.keras.models.Sequential(layers)
        #########################################################################
    else:
        raise Exception('Wrong NETWORK_TYPE!')

# The set of functions to clean the results.
# Such pre-processing allows to obtain more readible log-plots
########################################################################################
def remove_inf_nan(arr):
    return np.asarray([e for e in np.ravel(arr) if not np.isinf(e) and not np.isnan(e)])

def replace_zero_w_nan(arr):
    shape = np.asarray(arr).shape
    return np.asarray([np.nan if e == 0. else e for e in np.ravel(arr)]).reshape(shape)
########################################################################################

def run():
    # total_quartiles = [[] for _ in range(NUMBER_OF_X_RUNS)]
    total_quartiles = [[[[] for _ in range(len(depth_array))] for _ in range(len(c_array))]
        for _ in range(NUMBER_OF_X_RUNS)]

    # Generate all inputs and ensurre that they are not repeated
    x_array = []
    for xi in range(NUMBER_OF_X_RUNS):
        is_new_x = False
        while not is_new_x:
            x = np.random.normal(loc=0., scale=1., size=[1] + list(INPUT_SHAPE))
            norm = np.linalg.norm(x)
            if norm != 0:
                x = x / np.linalg.norm(x)

            # Check if such input was already created
            is_new_x = True
            for another_x in x_array:
                if np.array_equal(x, another_x):
                    is_new_x = False
                    break
        x_array.append(x)

    # Running the experiment
    for xi in range(NUMBER_OF_X_RUNS):
        x = x_array[xi]
        for ci, c in enumerate(c_array):
            for di, depth in enumerate(depth_array):
                # Getting all results
                ################################################################################################
                ################################################################################################
                norm_list = []

                # Get gradients for all runs
                ########################################################################################
                for run in range(NUMBER_OF_RUNS):
                    model = get_net(depth=depth, c=c)

                    with tf.GradientTape() as tape:
                        pred = model(x)
                        trainable_weights = [w for w in model.trainable_weights if 'kernel' in w.name]
                        grads = tape.gradient(pred, trainable_weights)
                    tf.keras.backend.clear_session()

                    all_grads = []
                    for layer_grad in grads:
                        # Adds gradients only for the weights where the gradient was not zero
                        # to include only the weights that contribute to the output.
                        # Assumes that the gradient is zero only for the weights that are not chosen by the maxout
                        all_grads.extend([g for g in layer_grad.numpy().ravel() if g!= 0])
                    norm_list.extend(np.asarray(all_grads)**2)
                ########################################################################################
                norm_list = remove_inf_nan(norm_list)
                if len(norm_list) > 0:
                    quartiles = replace_zero_w_nan(np.quantile(norm_list, [0.25, 0.5, 0.75]))
                # This case occurs if all results are zero or infinity
                else:
                    quartiles = np.asarray([np.nan, np.nan, np.nan])
                ################################################################################################
                ################################################################################################

                total_quartiles[xi][ci][di] = quartiles

    if NETWORK_TYPE == 'fc':
        coef = INPUT_SHAPE[-1]
    elif NETWORK_TYPE == 'cnn':
        coef = np.prod(INPUT_SHAPE)
    else:
        raise Exception('Wrong NETWORK_TYPE')

    quartiles = coef * np.asarray(total_quartiles)
    # Remove values where not all quartiles are available
    for xi in range(NUMBER_OF_X_RUNS):
        for ci in range(len(c_array)):
            for di in range(len(depth_array)):
                for qi in range(3):
                    # Check for nan
                    if np.isnan(quartiles[xi][ci][di][qi]):
                        for xi2 in range(NUMBER_OF_X_RUNS):
                            for qi2 in range(3):
                                quartiles[xi2][ci][di][qi2] = np.nan

    # Plotting the results
    plot_quartiles(x=depth_array, quartiles=quartiles, c_array=c_array,
        ylabel='$n_0 (\partial \mathcal{N} / \partial W_{i,k\',j})^2$')

def plot_quartiles(x, quartiles, c_array, ylabel):
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=44)
    ax.tick_params(axis='both', which='minor', labelsize=44)

    ax.set_yscale('log')
    plt.plot(x, [1 for _ in range(len(x))], linewidth=2, color='black')

    for i, (yi) in enumerate(quartiles):
        for ci, c in enumerate(c_array):
            q1 = np.asarray([yc[0] for yc in yi[ci]])
            q2 = np.asarray([yc[1] for yc in yi[ci]])
            q3 = np.asarray([yc[2] for yc in yi[ci]])
            if i == 0:
                plt.plot(x, q2, linewidth=3, color=c_to_color[c], label=c)
            else:
                plt.plot(x, q2, linewidth=3, color=c_to_color[c])
            plt.fill_between(x, q1, q3, facecolor=c_to_color[c], alpha=0.075)

    plt.ylabel(ylabel, size=50)
    plt.xlabel("Depth", size=50)
    xticks = [depth_array[i] for i in range(len(depth_array)) if i % 2 == 0]
    ax.set_xticks(xticks)
    plt.xlim(xmin=depth_array[0], xmax=depth_array[-1])

    plt.legend(loc='center right', title='Value of $c$', fontsize=40, title_fontsize=50, bbox_to_anchor=(1.4, 0.5))
    plt.savefig(f'quartiles.png', bbox_inches='tight')

def main(args):
    run()

if __name__ == "__main__":
    main(sys.argv[1:])
