# mish.py

import tensorflow as tf

# Seel also: https://arxiv.org/abs/2107.12461
@tf.function
def mish(x):
    x = tf.convert_to_tensor(x) #Added this line
    return tf.math.multiply(x, tf.math.tanh(tf.math.softplus(x)))
