import tensorflow as tf
import scipy

def conv2d(x, k_h, k_w, channels_in, channels_out, stride, name="conv"):
    with tf.name_scope(name):
        # Define weights
        w = tf.Variable(tf.truncated_normal([k_h,k_w, channels_in, channels_out], stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")    
        # Convolution
        #conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')    
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID')    
        # Relu
        activation = tf.nn.relu(conv + b)
        # Add summaries for helping debug
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activation", activation)
        return activation

def max_pool(x, k_h, k_w, S, name="maxpool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1],strides=[1, S, S, 1], padding='SAME')

def fc_layer(x, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")    
        activation = tf.nn.relu(tf.matmul(x, w) + b)
        # Add summaries for helping debug
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activation", activation)
        return activation

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#def conv2d(x, W, stride):
  #return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

# Create placeholders (I/O for our model/graph)
x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

#first convolutional layer
#W_conv1 = weight_variable([5, 5, 3, 24])
#b_conv1 = bias_variable([24])

#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)
conv1 = conv2d(x_image, 5, 5, 3, 24, 2,"conv1")

#second convolutional layer
#W_conv2 = weight_variable([5, 5, 24, 36])
#b_conv2 = bias_variable([36])

#h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
conv2 = conv2d(conv1, 5, 5, 24, 36, 2,"conv2")

#third convolutional layer
#W_conv3 = weight_variable([5, 5, 36, 48])
#b_conv3 = bias_variable([48])

#h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
conv3 = conv2d(conv2, 5, 5, 36, 48, 2,"conv3")

#fourth convolutional layer
#W_conv4 = weight_variable([3, 3, 48, 64])
#b_conv4 = bias_variable([64])

#h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)
conv4 = conv2d(conv3, 3, 3, 48, 64, 1, "conv4")

#fifth convolutional layer
#W_conv5 = weight_variable([3, 3, 64, 64])
#b_conv5 = bias_variable([64])

#h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
conv5 = conv2d(conv4, 3, 3, 64, 64, 1, "conv5")

#FCL 1
#W_fc1 = weight_variable([1152, 1164])
#b_fc1 = bias_variable([1164])

# Needs calculation... (-1 means any batch size)
conv5_flat = tf.reshape(conv5, [-1, 1152])
#h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
fc1 = fc_layer(conv5_flat, 1152, 1164, "fc1")

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Add dropout to the fully connected layer
fc1_drop = tf.nn.dropout(fc1, 0.8)

#FCL 2
#W_fc2 = weight_variable([1164, 100])
#b_fc2 = bias_variable([100])

#h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
fc2 = fc_layer(fc1_drop, 1164, 100, "fc2")

#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
fc2_drop = tf.nn.dropout(fc2, 0.8)

#FCL 3
#W_fc3 = weight_variable([100, 50])
#b_fc3 = bias_variable([50])

#h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
fc3 = fc_layer(fc2_drop, 100, 50, "fc3")

#h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
fc3_drop = tf.nn.dropout(fc3, 0.8)

#FCL 4
#W_fc4 = weight_variable([50, 10])
#b_fc4 = bias_variable([10])

#h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
fc4 = fc_layer(fc3_drop, 50, 10, "fc4")

#h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
fc4_drop = tf.nn.dropout(fc4, 0.8)

#Output
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable([1])

# Normalize output between -2..2
# https://www.wolframalpha.com/input/?i=atan(x)
y = tf.multiply(tf.atan(tf.matmul(fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
