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

def output_layer(x, channels_in, channels_out, name="output"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")    
        activation = tf.matmul(x, w) + b
        # Add summaries for helping debug
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activation", activation)
        return activation    

def bound_layer(val_in, bound_val, name="bounding_func"):
    with tf.name_scope(name):        
        activation = tf.multiply(tf.atan(val_in), bound_val)
        # Add summaries for helping debug        
        tf.summary.histogram("val_in", val_in)
        tf.summary.histogram("activation", activation)
        return activation        


# Create placeholders (I/O for our model/graph)
x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

# CONV 1
conv1 = conv2d(x_image, 5, 5, 3, 24, 2,"conv1")

# CONV 2
conv2 = conv2d(conv1, 5, 5, 24, 36, 2,"conv2")

# CONV 3
conv3 = conv2d(conv2, 5, 5, 36, 48, 2,"conv3")

# CONV 4
conv4 = conv2d(conv3, 3, 3, 48, 64, 1, "conv4")

# CONV 5
conv5 = conv2d(conv4, 3, 3, 64, 64, 1, "conv5")

# Fully Connect 1
# Needs calculation... (-1 means any batch size)
conv5_flat = tf.reshape(conv5, [-1, 1152])
fc1 = fc_layer(conv5_flat, 1152, 1164, "fc1")
# Add dropout to the fully connected layer
fc1_drop = tf.nn.dropout(fc1, 0.8)

# Fully Connect 2
fc2 = fc_layer(fc1_drop, 1164, 100, "fc2")
# Add dropout to the fully connected layer
fc2_drop = tf.nn.dropout(fc2, 0.8)

# Fully Connect 3
fc3 = fc_layer(fc2_drop, 100, 50, "fc3")
# Add dropout to the fully connected layer
fc3_drop = tf.nn.dropout(fc3, 0.8)

# Fully Connect 4
fc4 = fc_layer(fc3_drop, 50, 10, "fc4")
# Add dropout to the fully connected layer
fc4_drop = tf.nn.dropout(fc4, 0.8)

#Output
out_layer = output_layer(fc4_drop, 10, 1, "output_layer")

# Bounding output and scale between -2..2
# https://www.wolframalpha.com/input/?i=atan(x)
y = bound_layer(out_layer,2)
