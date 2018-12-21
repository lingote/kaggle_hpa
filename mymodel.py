import tensorflow as tf
import numpy as np
from dataprep import get_data, get_img_array

def create_model(input_img, n_out):
    layer_name = 'final_layer_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('dense_layer'):
            logits = tf.layers.dense(input_img, n_out,
                                     kernel_initializer=tf.initializers.he_normal(),
                                     bias_initializer=tf.initializers.zeros,
                                     activation='sigmoid', trainable=True)
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                input_img.shape.as_list(), stddev=0.001
            )
            layer_weights = tf.Variable(initial_value, name='final_weights')

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([n_out]))

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(input_img, layer_weights) + layer_biases

    final_tensor = tf.nn.sigmoid(logits, name='final_tensor_name')