import tensorflow as tf
import numpy as np
import tensornets as tnets
from dataprep import get_datalist, get_img_array

def create_model(input_img, n_out):
    layer_name = 'final_layer_ops'
    with tf.variable_scope(layer_name) as scope:
        inputs = tf.placeholder(tf.float32, [None, 299,299,3])
        outputs = tf.placeholder(tf.float32, [None, 28])
        logits = tnets.Inception4(inputs, is_training=True, classes=28)
        model =tf.identity(logits, name='logits')
        loss = tf.losses.sigmoid_cross_entropy(outputs, model)
        train =tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
        final_tensor = tf.nn.sigmoid(logits, name='final_tensor_name')

def runmodel(model):
    with tf.Session() as sess:
        sess.run(model.pretrained())
        