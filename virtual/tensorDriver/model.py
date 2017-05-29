import tensorflow as tf
import model_util as util

class DrivingModel(object):
    def __init__(self, input = None, use_placeholder = True):
        self.__x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='IMAGE_IN')
        self.__y_ = tf.placeholder(tf.float32, shape=[None, 1], name='LABEL_IN')
        self.__dropout_prob = tf.placeholder(tf.float32, name='drop_prob')
        self.__use_placeholder = use_placeholder

        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 24, 2, "conv1", True)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 24, 2, "conv1", True)

        # CONV 2
        self.__conv2 = util.conv2d(self.__conv1, 5, 5, 24, 36, 2, "conv2")

        # CONV 3
        self.__conv3 = util.conv2d(self.__conv2, 5, 5, 36, 48, 2, "conv3")

        # CONV 4
        self.__conv4 = util.conv2d(self.__conv3, 3, 3, 48, 64, 1, "conv4")

        # CONV 5
        self.__conv5 = util.conv2d(self.__conv4, 3, 3, 64, 64, 1, "conv5")

        # Fully Connect 1
        # Needs calculation... (-1 means any batch size)
        with tf.name_scope('reshape_conv5'):
            self.__conv5_flat = tf.reshape(self.__conv5, [-1, 1152])

        self.__fc1 = util.fc_layer(self.__conv5_flat, 1152, 1164, "fc1")
        # Add dropout to the fully connected layer
        self.__fc1_drop = tf.nn.dropout(self.__fc1, self.__dropout_prob)

        # Fully Connect 2
        self.__fc2 = util.fc_layer(self.__fc1_drop, 1164, 100, "fc2")
        # Add dropout to the fully connected layer
        self.__fc2_drop = tf.nn.dropout(self.__fc2, self.__dropout_prob)

        # Fully Connect 3
        self.__fc3 = util.fc_layer(self.__fc2_drop, 100, 50, "fc3")
        # Add dropout to the fully connected layer
        self.__fc3_drop = tf.nn.dropout(self.__fc3, self.__dropout_prob)

        # Fully Connect 4
        self.__fc4 = util.fc_layer(self.__fc3_drop, 50, 10, "fc4")
        # Add dropout to the fully connected layer
        self.__fc4_drop = tf.nn.dropout(self.__fc4, self.__dropout_prob)

        # Output
        self.__y = util.output_layer(self.__fc4_drop, 10, 1, "output_layer")

    @property
    def output(self):
        return self.__y

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__y_
        else:
            return None

    @property
    def dropout_control(self):
        return self.__dropout_prob
