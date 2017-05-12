import argparse
import os
import tensorflow as tf
from driving_data import HandleData
import model
import model_util as util

# Example: python train.py --input=DrivingData.h5 --gpu=0 --checkpoint_dir=save/model.ckpt
parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('--input_list', type=str, required=False, default='FileList.txt', help='Training list of TFRecord files')
parser.add_argument('--input_val', type=str, required=False, default='', help='Validation TFRecord file')
parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU number (-1) for CPU')
parser.add_argument('--checkpoint_dir', type=str, required=False, default='', help='Load checkpoint')
parser.add_argument('--logdir', type=str, required=False, default='./logs', help='Tensorboard log directory')
parser.add_argument('--savedir', type=str, required=False, default='./save', help='Tensorboard checkpoint directory')
parser.add_argument('--epochs', type=int, required=False, default=600, help='Number of epochs')
parser.add_argument('--batch', type=int, required=False, default=400, help='Batch size')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='Initial learning rate')
args = parser.parse_args()

def create_input_graph(list_files, num_epochs, batch_size):
    with tf.name_scope('input_handler'):
        filename_queue = tf.train.string_input_producer(list_files, num_epochs=num_epochs)

        # Read files from TFRecord list
        image, label = util.read_decode_tfrecord_list(filename_queue)

        # Shuffle examples
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return images, labels

def train_network(input_list, input_val_hdf5, gpu, pre_trained_checkpoint, epochs, batch_size, logs_path, save_dir):

    # Create log directory if it does not exist
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Set enviroment variable to set the GPU to use
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('Set tensorflow on CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Define number of epochs and batch size, where to save logs, etc...
    iter_disp = 10
    start_lr = args.learning_rate

    # Avoid allocating the whole memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    # Regularization value
    L2NormConst = 0.001

    # Get all model "parameters" that are trainable
    train_vars = tf.trainable_variables()

    # Loss is mean squared error plus l2 regularization
    # model.y (Model output), model.y_(Labels)
    # tf.nn.l2_loss: Computes half the L2 norm of a tensor without the sqrt
    # output = sum(t ** 2) / 2
    with tf.name_scope("MSE_Loss_L2Reg"):
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

    # Add model accuracy
    with tf.name_scope("Loss_Validation"):
        loss_val = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))

    # Solver configuration
    with tf.name_scope("Solver"):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = start_lr
        # decay every 10000 steps with a base of 0.96
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   1000, 0.9, staircase=True)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Initialize all random variables (Weights/Bias)
    sess.run(tf.global_variables_initializer())

    # Load checkpoint if needed
    if pre_trained_checkpoint:
        # Load tensorflow model
        print("Loading pre-trained model: %s" % args.checkpoint_dir)
        # Create saver object to save/load training checkpoint
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, args.checkpoint_dir)
    else:
        # Just create saver for saving checkpoints
        saver = tf.train.Saver(max_to_keep=None)

    # Monitor loss, learning_rate, global_step, etc...
    tf.summary.scalar("loss_train", loss)
    tf.summary.scalar("loss_val", loss_val)
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("global_step", global_step)
    # merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Configure where to save the logs for tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Get file list
    list_tfrecord_files = HandleData.get_list_from_file(input_list)

    # Create the graph input part (Responsible to load files, do augmentations, etc...)
    create_input_graph(list_tfrecord_files, epochs, batch_size)



if __name__ == "__main__":
    # Call function that implement the auto-pilot
    train_network(args.input_list, args.input_val, args.gpu,
                  args.checkpoint_dir, args.epochs, args.batch, args.logdir, args.savedir)
