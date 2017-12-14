from datetime import datetime
import math
import time
import numpy as np
import dataset
import tensorflow.python.platform
import tensorflow as tf
import layers as L
from tensorflow.contrib import rnn
from logging import getLogger
from twolayerlstm import *
from ops import *

flags = tf.app.flags

# network
flags.DEFINE_string("model", "pixel_cnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("hidden_dims", 16, "dimesion of hidden states of LSTM or Conv layers")
flags.DEFINE_integer("recurrent_length", 7, "the length of LSTM or Conv layers")
flags.DEFINE_integer("out_hidden_dims", 32, "dimesion of hidden states of output Conv layers")
flags.DEFINE_integer("out_recurrent_length", 2, "the length of output Conv layers")
flags.DEFINE_boolean("use_residual", True, "whether to use residual connections or not")
# flags.DEFINE_boolean("use_dynamic_rnn", False, "whether to use dynamic_rnn or not")

# training
flags.DEFINE_integer("max_epoch", 100000, "# of step in an epoch")
flags.DEFINE_integer("test_step", 100, "# of step to test a model")
flags.DEFINE_integer("save_step", 1000, "# of step to save a model")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")

# data
flags.DEFINE_string("data", "mnist", "name of dataset [mnist, cifar]")
flags.DEFINE_string("data_dir", "data", "name of data directory")
flags.DEFINE_string("sample_dir", "samples", "name of sample directory")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_boolean("display", False, "whether to display the training results or not")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 123, "random seed for python")

conf = flags.FLAGS

l = {}
logger = logging.getLogger(__name__)
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def build(input_tensor, n_classes=1000, rgb_mean=None, training=True):
    # assuming 224x224x3 input_tensor

    # define image mean
    if rgb_mean is None:
        rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
    mu = tf.constant(rgb_mean, name="rgb_mean")
    keep_prob = 0.5

    # subtract image mean
    input_mean_centered = tf.subtract(input_tensor, mu, name="input_mean_centered")

    # block 1 -- outputs 112x112x64
    conv1_1 = L.conv(input_mean_centered, name="conv1_1", kh=3, kw=3, n_out=64)
    conv1_2 = L.conv(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64)
    pool1 = L.pool(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = L.conv(pool1, name="conv2_1", kh=3, kw=3, n_out=128)
    conv2_2 = L.conv(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128)
    pool2 = L.pool(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = L.conv(pool2, name="conv3_1", kh=3, kw=3, n_out=256)
    conv3_2 = L.conv(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256)
    pool3 = L.pool(conv3_2, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = L.conv(pool3, name="conv4_1", kh=3, kw=3, n_out=512)
    conv4_2 = L.conv(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512)
    conv4_3 = L.conv(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512)
    pool4 = L.pool(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = L.conv(pool4, name="conv5_1", kh=3, kw=3, n_out=512)
    conv5_2 = L.conv(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512)
    conv5_3 = L.conv(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512)
    # pool5 = L.pool(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)


    pool2_max = L.pool(pool2, name="pool2_max", kh=2, kw=2, dw=2, dh=2)
    pool2_max_conv = L.conv(pool2_max, name="pool2_max_conv", kh=3, kw=3, n_out=512)

    pool3_conv = L.conv(pool3, name="pool3_conv", kh=3, kw=3, n_out=512)

    conv4_3_conv = L.conv(conv4_3, name="conv4_3_conv", kh=3, kw=3, n_out=512)

    conv5_3_conv = L.conv(conv5_3, name="conv5_3_conv", kh=3, kw=3, n_out=512)
    conv5_3_conv_upsampling = tf.image.resize_images(conv5_3_conv,[28,28],method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

    concatTensor = tf.concat([pool2_max_conv, pool3_conv], 0)
    concatTensor = tf.concat([concatTensor, conv4_3_conv], 0)
    concatTensor = tf.concat([concatTensor, conv5_3_conv_upsampling], 0)
    concatTensor_conv = L.conv(concatTensor, name="concatTensor_conv", kh=3, kw=3, n_out=128)

    hidden_layer_size = 30
    input_size = 8
    target_size = 10
    rnn = LSTM_cell(input_size, hidden_layer_size, target_size)
    # scope = "conv_inputs"
    # logger.info("Building %s" % scope)
    # Getting all outputs from rnn
    outputs = rnn.get_outputs()

    # Getting final output through indexing after reversing
    last_output = outputs[-1]

    # main reccurent layers
    # l_hid = concatTensor_conv
    # for idx in range(conf.recurrent_length):
    #     scope = 'LSTM%d' % idx
    #     l[scope] = l_hid = diagonal_bilstm(l_hid, conf, scope=scope)


    # cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[28, 28],
    #                                      # size of input feeding into network (needed for the zero state)
    #                                      kernel_shape=[3, 3],  # for a 3 by 3 conv
    #                                      output_channels=128)  # number of feature maps
    ##########################################################################
    # Now running the dynamic rnn
    ##########################################################################
    # (outputs, state) = tf.nn.dynamic_rnn(cell, concatTensor_conv, time_major=False, dtype=tf.float32)

    ##########################################################################
    # Now treat the hidden state out of the conv lstm as the new image
    ##########################################################################
    x_image = state[0]
# tf.contrib.rnn.ConvLSTMCell(conv_ndims,input_shape,output_channels, kernel_shape,use_bias=True,skip_connection=False,forget_bias=1.0,
#                                 initializers=None,    name='conv_lstm_cell')
    # flatten
    # flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    # net = tf.reshape(net, [-1, flattened_shape], name="flatten")

    # fully connected
    # net = L.fully_connected(net, name="fc6", n_out=4096)
    # net = tf.nn.dropout(net, keep_prob)
    # net = L.fully_connected(net, name="fc7", n_out=4096)
    # net = tf.nn.dropout(net, keep_prob)
    # net = L.fully_connected(net, name="fc8", n_out=n_classes)
    return net

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [10, 224, 224, 3])
    net = build(x)