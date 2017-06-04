import tensorflow as tf
import model_util as util

class DrivingModel(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True):
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

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn)

        # CONV 2
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 24, 36, 2, "conv2")
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn)

        # CONV 3
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 36, 48, 2, "conv3")
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn)

        # CONV 4
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 48, 64, 1, "conv4")
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn)

        # CONV 5
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5")
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn)

        # Fully Connect 1
        # Needs calculation... (-1 means any batch size)
        with tf.name_scope('reshape_conv5'):
            self.__conv5_flat = tf.reshape(self.__conv5_act, [-1, 1152])

        self.__fc1 = util.linear_layer(self.__conv5_flat, 1152, 1164, "fc1")
        #self.__fc1_bn = util.batch_norm(self.__fc1, training_mode)
        self.__fc1_act = util.relu(self.__fc1)
        # Add dropout to the fully connected layer
        self.__fc1_drop = tf.nn.dropout(self.__fc1_act, self.__dropout_prob)

        # Fully Connect 2
        self.__fc2 = util.linear_layer(self.__fc1_drop, 1164, 100, "fc2")
        #self.__fc2_bn = util.batch_norm(self.__fc2, training_mode)
        self.__fc2_act = util.relu(self.__fc2)
        # Add dropout to the fully connected layer
        self.__fc2_drop = tf.nn.dropout(self.__fc2_act, self.__dropout_prob)

        # Fully Connect 3
        self.__fc3 = util.linear_layer(self.__fc2_drop, 100, 50, "fc3")
        #self.__fc3_bn = util.batch_norm(self.__fc3, training_mode)
        self.__fc3_act = util.relu(self.__fc3)
        # Add dropout to the fully connected layer
        self.__fc3_drop = tf.nn.dropout(self.__fc3_act, self.__dropout_prob)

        # Fully Connect 4
        self.__fc4 = util.linear_layer(self.__fc3_drop, 50, 10, "fc4")
        #self.__fc4_bn = util.batch_norm(self.__fc4, training_mode)
        self.__fc4_act = util.relu(self.__fc4)
        # Add dropout to the fully connected layer
        self.__fc4_drop = tf.nn.dropout(self.__fc4_act, self.__dropout_prob)

        # Output
        self.__y = util.linear_layer(self.__fc4_drop, 10, 1, "output_layer")

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

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


# http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
# http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf
# https://stackoverflow.com/questions/35980044/getting-the-output-shape-of-deconvolution-layer-using-tf-nn-conv2d-transpose-in
# https://arxiv.org/pdf/1411.4038.pdf
class DrivingModelAutoEncoder(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True):
        self.__x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='IMAGE_IN')
        self.__y_ = tf.placeholder(tf.float32, shape=[None, 1], name='LABEL_IN')
        self.__dropout_prob = tf.placeholder(tf.float32, name='drop_prob')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 66x200x3 after CONV 5x5 P:0 S:2 H_out: 1 + (66-5)/2 = 31, W_out= 1 + (200-5)/2=98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 24, 2, "conv1", True, do_summary = False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 24, 2, "conv1", True, do_summary = False)

        self.__conv1_act = util.relu(self.__conv1, do_summary = False)

        # CONV2: Input 31x98x24 after CONV 5x5 P:0 S:2 H_out: 1 + (31-5)/2 = 14, W_out= 1 + (200-5)/2=47
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 24, 36, 2, "conv2", do_summary = False)
        self.__conv2_act = util.relu(self.__conv2, do_summary = False)

        # CONV3: Input 14x47x36 after CONV 5x5 P:0 S:2 H_out: 1 + (14-5)/2 = 5, W_out= 1 + (47-5)/2=22
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 36, 48, 2, "conv3", do_summary = False)
        self.__conv3_act = util.relu(self.__conv3, do_summary = False)

        # CONV4: Input 5x22x48 after CONV 3x3 P:0 S:1 H_out: 1 + (5-3)/1 = 3, W_out= 1 + (22-3)/1=20
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 48, 64, 1, "conv4", do_summary = False)
        self.__conv4_act = util.relu(self.__conv4, do_summary = False)

        # CONV5: Input 3x20x64 after CONV 3x3 P:0 S:1 H_out: 1 + (3-3)/1 = 1, W_out= 1 + (20-3)/1=18
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5", do_summary = False)
        self.__conv5_act = util.relu(self.__conv5, do_summary = False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (3, 20), 64, 64, 1, name="dconv1", do_summary = False)
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out, do_summary = False)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act, (3, 3), (5, 22), 64, 48, 1, name="dconv2", do_summary = False)
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out, do_summary = False)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act, (5, 5), (14, 47), 48, 36, 2, name="dconv3", do_summary = False)
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out, do_summary = False)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act, (5, 5), (31, 98), 36, 24, 2, name="dconv4", do_summary = False)
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out, do_summary = False)

        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (66, 200), 24, 64, 2, name="dconv5", do_summary = False)
        self.__conv_t1_out_act = util.relu(self.__conv_t1_out, do_summary = False)

        # Just adapt volume to from 66x200x64 to 66x200x3 (Output)
        # This adaptation is actually not needed you could change the filter dimensions on conv1_transp to 64,3
        self.__y_out = util.conv2d(self.__conv_t1_out_act, 1, 1, 64, 3, 1, name="conv_out", pad='SAME', do_summary = False)
        self.__y = util.sigmoid(self.__y_out, do_summary = False)


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

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act

