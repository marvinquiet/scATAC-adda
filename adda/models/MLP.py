from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('MLP')
def MLP(inputs, n_clusters, scope='MLP', is_training=True, reuse=False,
        dropout_keep_prob=0.9):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
                    ))
            stack.enter_context(slim.arg_scope([slim.dropout], keep_prob=dropout_keep_prob))
            net = slim.fully_connected(net, 64, scope='fc1')
            net = slim.dropout(net, is_training=is_training)
            layers['fc1'] = net
            net = slim.fully_connected(net, 16, scope='fc2')
            net = slim.dropout(net, is_training=is_training)
            layers['fc2'] = net
            net = slim.fully_connected(net, n_clusters, activation_fn=None, scope='fc3')
            layers['fc3'] = net
    return net, layers
MLP.num_channels = 1
MLP.mean = None
MLP.bgr = False
