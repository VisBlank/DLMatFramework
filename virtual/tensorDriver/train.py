import os
import tensorflow as tf
from driving_data import HandleData
import model

LOGDIR = './save'

sess = tf.InteractiveSession()

# Regularization value
L2NormConst = 0.001

# Get all model "parameters" that are trainable
train_vars = tf.trainable_variables()

# Loss is mean squared error plus l2 regularization
# model.y (Model output), model.y_(Labels)
# tf.nn.l2_loss: Computes half the L2 norm of a tensor without the sqrt
# output = sum(t ** 2) / 2
with tf.name_scope("MeanSquared_Loss_L2_Reg"):
    loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n(
        [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# Add model accuracy
with tf.name_scope("Loss_Validation"):
    loss_val = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))

# Solver configuration
with tf.name_scope("Solver"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Initialize all random variables (Weights/Bias)
sess.run(tf.global_variables_initializer())

# Monitor loss
tf.summary.scalar("loss_train", loss)
tf.summary.scalar("loss_val", loss_val)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Create saver object to save training checkpoint
saver = tf.train.Saver()

# Configure where to save the logs for tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# Define number of epochs and batch size
epochs = 600
batch_size = 100
iter_disp = 10

data = HandleData()

# For each epoch
for epoch in range(epochs):
    for i in range(int(data.get_num_images() / batch_size)):
        # Get training batch
        xs, ys = data.LoadTrainBatch(batch_size)

        # Send batch to tensorflow graph
        train_step.run(feed_dict={model.x: xs, model.y_: ys})

        # Display some information each x iterations
        if i % iter_disp == 0:
            # Get validation batch
            xs, ys = data.LoadValBatch(batch_size)
            # loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys})
            loss_value = loss_val.eval(feed_dict={model.x: xs, model.y_: ys})
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

        # write logs at every iteration
        summary = merged_summary_op.eval(feed_dict={model.x: xs, model.y_: ys})
        summary_writer.add_summary(summary, epoch * batch_size + i)

        # Save checkpoint after each epoch
        if i % batch_size == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
    # Shuffle data at each epoch end
    print("Shuffle data")
    data.shuffleData()

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
