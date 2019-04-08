"""
Tensorflow script for training a model from preprocessed voice data.

Usage:

with downloaded data:

python3 train_tensorflow.py --path_to_data=/path_to_downloaded data

OR with randomly generated data:

python3 train_tensorflow.py --random_data=True
"""

import argparse
import bcolz
import numpy as np
import os
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim

# config
batch_size = 32
dropout_rate = 0.2
learning_rate = 0.0001
num_epochs = 100

input_features = 16000  # X.shape is [-1, input_features]
num_x_subsets = 7  # due to memory, we've split the X into these many subsets
num_classes = 12

# paths
default_path_to_data = os.path.join('data', 'main', 'preprocessed')


# define the bcolz array saving functions
def bcolz_save(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()


def bcolz_load(file_name):
    return bcolz.open(file_name)[:]


def load_data(data_path, logger):
    """
    Function for loading the X and y for the voice recognition challenge.
    Assumes previous steps from the jupyter notebooks were followed.
    :param data_path: a path to the preprocessed bcolz data sets
    :param logger: a logging.Logger() instance
    :return: X and y for all 3 subsets - train, cv and test.
    """

    # load the y
    logger.info('\nLoading the y sets ...')
    training_y = bcolz_load(os.path.join(data_path, 'train_y.bc'))
    validation_y = bcolz_load(os.path.join(data_path, 'cv_y.bc'))
    testing_y = bcolz_load(os.path.join(data_path, 'test_y.bc'))

    # report shapes
    logger.info('y train shape: {}'.format(training_y.shape))
    logger.info('y cv shape: {}'.format(validation_y.shape))
    logger.info('y test shape: {}'.format(testing_y.shape))

    # reload the Test & CV X
    logger.info('\nLoading the X sets ...')
    validation_x = bcolz_load(data_path + os.path.sep + 'cv_X.bc')
    testing_x = bcolz_load(data_path + os.path.sep + 'test_X.bc')

    # reload the Train X
    logger.info('Loading the X train subsets ...')
    training_xs = []

    for i in range(num_x_subsets):
        # load each subset
        train_subset = bcolz_load(data_path + os.path.sep + 'train_X' + str(i + 1) + '.bc')

        logger.info('X subset #{}s shape: {}'.format(i + 1, train_subset.shape))
        training_xs.append(train_subset)

    # concatenate the Train Xs back into one matrix
    logger.info('\nConcatenating the X train subsets...')
    training_x = np.concatenate(tuple(training_xs))

    # expand the dimensions
    logger.info('\nExpanding the dimension of the combined X tensor for 1D '
                'convolutional operations ...')
    training_x = np.expand_dims(training_x, axis=2)
    validation_x = np.expand_dims(validation_x, axis=2)
    testing_x = np.expand_dims(testing_x, axis=2)

    # report final X shapes
    logger.info('Expanded X train shape: {}'.format(training_x.shape))
    logger.info('Expanded X cv shape: {}'.format(validation_x.shape))
    logger.info('Expanded X test shape: {}\n'.format(testing_x.shape))

    return (training_x, training_y,
            validation_x, validation_y,
            testing_x, testing_y)


def generate_random_data(logger):
    """
    Function for generating random X and y for the voice recognition challenge.
    :param logger: a logging.Logger() instance
    :return: X and y for all 3 subsets - train, cv and test.
    """

    # generate the y
    logger.info('\nGenerating the y sets ...')
    training_y = np.random.rand(22176, num_classes)
    validation_y = np.random.rand(3051, num_classes)
    testing_y = np.random.rand(3087, num_classes)

    # report shapes
    logger.info('y train shape: {}'.format(training_y.shape))
    logger.info('y cv shape: {}'.format(validation_y.shape))
    logger.info('y test shape: {}'.format(testing_y.shape))

    # generate the X
    logger.info('\nGenerating the X sets ...')
    validation_x = np.random.rand(3051, input_features, 1)
    testing_x = np.random.rand(3087, input_features, 1)
    training_x = np.random.rand(22176, input_features, 1)

    # report shapes
    logger.info('X train shape: {}'.format(training_x.shape))
    logger.info('X cv shape: {}'.format(validation_x.shape))
    logger.info('X test shape: {}'.format(testing_x.shape))

    return (training_x, training_y,
            validation_x, validation_y,
            testing_x, testing_y)


def create_logger(name='main', level='INFO', tf_log_level='3'):
    """
    Set log levels for both logging and tf logging.
    :param name: a string to name the logger after
    :param tf_log_level: a string representing preferred tf log level
    :param level: a string representing the main logger's preferred level
    :param tf_log_level: a string representing preferred tf log level
    :return: a logging.Logger() with proper handlers
    """
    # only show tf error logs, won't work in all tf versions
    # e.g. it's ok to get certain FutureWarnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level

    # main logger
    a_logger = logging.Logger(name)
    a_logger.setLevel(level)

    # add log handler for console output
    console_handler = logging.StreamHandler()
    a_logger.addHandler(console_handler)

    return a_logger


def create_parser():
    """
    Create a simple command line argument parser for the data directory.
    :return: an argparse.ArgumentParser() instance
    """
    a_parser = argparse.ArgumentParser()

    # for data path
    a_parser.add_argument("--path_to_data",
                          type=str,
                          default=default_path_to_data,
                          help="Path to the data .bc files' parent directory")
    # else, for random data
    a_parser.add_argument("--random_data",
                          type=bool,
                          default=False,
                          help="Boolean marking whether random data is to be"
                               " used (when actual data has not been "
                               "donwloaded).")
    # tensorboard logdir
    a_parser.add_argument("--logdir",
                          type=str,
                          default="tf_logs",
                          help="String representing the logdir to which"
                               "a running instance of tensorboard has been"
                               "pointed to.")
    # tensorboard logdir
    a_parser.add_argument("--logrun",
                          type=str,
                          default="latest",
                          help="String representing the training run, which "
                               "will become a subdirectory in the logdir, read "
                               "by tensorboard for metrics visualizations.")

    return a_parser


def model_summary(logger):
    """
    Use tf.contrib.slim to get the equivalent of keras' model.summary()
    :param logger: a logging.Logger() instance
    """
    logger.info('\nModel summary ...')
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def show_update_ops(logger):
    """
    Show all update ops, used for debugging of the batchnorm.
    :param logger: a logging.Logger() instance
    """
    logger.info('\nUpdate ops:\n')
    for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
        logger.info(op)


def show_trainable_vars(logger):
    """
    Show all trainable variables, used for debugging of the batchnorm.
    :param logger: a logging.Logger() instance
    """
    logger.info('\nTrainable variables:\n')
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        logger.info(var)


def create_model_graph(logger):
    """
    Create a tf graph of the model, with appropriate nodes and operations.
    :param logger: a logging.Logger() instance
    :return: the inputs, labels and outputs operations & in_training controller
    """
    logger.info('\nConstructing the model\'s graph definition ...')

    # X input & y labels
    inputs = tf.placeholder(tf.float32, [None, input_features, 1])
    labels = tf.placeholder(tf.float32, [None, num_classes])

    logger.info('\nInput X shape: {}'.format(inputs.get_shape().as_list()))
    logger.info('Input y shape (labels): {}'.format(labels.get_shape().as_list()))

    # control training & evaluation mode (e.g. for drop-out)
    in_training = tf.placeholder(tf.bool)

    # batch normalization first
    x = tf.layers.batch_normalization(inputs=inputs, training=in_training)

    # main convolutional loops
    for i in range(9):

        x = tf.layers.conv1d(inputs=x, filters=8*(2**i), kernel_size=3, padding='same')
        logger.info('Conv{}a: {}'.format(i+1, x.get_shape().as_list()))
        x = tf.layers.batch_normalization(inputs=x, training=in_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv1d(inputs=x, filters=8*(2**i), kernel_size=3, padding='same')
        logger.info('Conv{}b: {}'.format(i+1, x.get_shape().as_list()))
        x = tf.layers.batch_normalization(inputs=x, training=in_training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling1d(inputs=x, pool_size=2, strides=2, padding='same')
        logger.info('Pool{}: {}'.format(i+1, x.get_shape().as_list()))

    convLast = tf.layers.conv1d(inputs=x, filters=1024, kernel_size=1, padding='same', activation=tf.nn.relu)
    logger.info('ConvLast: {}'.format(convLast.get_shape().as_list()))

    poolLast = tf.layers.max_pooling1d(inputs=convLast, pool_size=2, strides=2, padding='same')
    logger.info('PoolLast: {}'.format(poolLast.get_shape().as_list()))

    flatLast = tf.layers.flatten(inputs=poolLast)
    logger.info('Flat1: {}'.format(flatLast.get_shape().as_list()))

    hidden = tf.layers.dense(inputs=flatLast, units=1024, activation=tf.nn.relu)
    logger.info('Hidden: {}'.format(hidden.get_shape().as_list()))

    dropout = tf.layers.dropout(inputs=hidden, rate=dropout_rate, training=in_training)

    outputs = tf.layers.dense(inputs=dropout, units=num_classes, activation=tf.nn.softmax)
    logger.info('Output: {}'.format(outputs.get_shape().as_list()))

    return inputs, labels, outputs, in_training


if __name__ == '__main__':

    # add parser
    parser = create_parser()
    params = parser.parse_args()

    # start
    print('\nStarting the tensorflow training script...')

    # grab the user-specified path to data
    path_to_data = params.path_to_data

    # grab the boolean whether to use downloaded or random-generated data
    use_random_data = params.random_data

    # handle logging
    logger = create_logger()

    # handle data source
    if use_random_data:
        # generate random data
        data = generate_random_data(logger)

    else:
        # load the downloaded data
        data = load_data(path_to_data, logger)

    # unpack the data
    train_X, train_y, cv_X, cv_y, test_X, test_y = data

    # batch norm update ops (before declaring the model)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # prepare the model graph
    inputs, labels, outputs, in_training = create_model_graph(logger)

    # show the model
    model_summary(logger)

    # loss
    loss = tf.losses.softmax_cross_entropy(labels, outputs)

    # metrics
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # optimizer
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        train_op = tf.group([train_step, update_ops])

    # inspect the ops (looking for moving average in batchnorm update ops)
    show_update_ops(logger)
    show_trainable_vars(logger)

    # tensorboard ops
    with tf.name_scope('performance'):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    # session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # extract the tensorboard logdir from params
    logdir = params.logdir
    logrun = params.logrun

    # create log directory and run subdirectory, if not present
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(os.path.join(logdir, logrun)):
        os.mkdir(os.path.join(logdir, logrun))

    summary_writer = tf.summary.FileWriter(os.path.join(logdir, logrun),
                                           graph=tf.get_default_graph())

    # train
    for epoch in range(num_epochs + 1):

        # track metrics per epoch
        epoch_loss = 0
        epoch_accuracy = 0

        # calculate steps per epoch based on batch size
        num_batches = int(len(train_X) / batch_size)

        # shuffle data after each epoch before splitting into batches
        indices = np.arange(train_X.shape[0])
        np.random.shuffle(indices)

        shuffled_train_X = train_X[indices]
        shuffled_train_y = train_y[indices]

        train_X_batches = np.split(shuffled_train_X, num_batches)
        train_y_batches = np.split(shuffled_train_y, num_batches)

        # train all batches
        for i, train_x_batch in enumerate(train_X_batches):

            # get batches
            input_batch = train_x_batch
            labels_batch = train_y_batches[i]

            # prepare the feed dict
            feed_dict = {inputs: input_batch,
                         labels: labels_batch,
                         in_training: True}

            # calculate batch loss
            _, batch_loss, summary = sess.run([train_op, loss,
                                               merged_summary_op],
                                              feed_dict=feed_dict)

            # add batch loss to total epoch loss
            epoch_loss += batch_loss

            # calculate the train accuracy on entire set
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            epoch_accuracy += train_accuracy

            # Run the optimization step
            train_op.run(feed_dict=feed_dict)

            # log every 25 batches
            if i % 25 == 0:
                logger.info('Batch: {} loss = {:5f}'.format(i+1, batch_loss))
                summary_writer.add_summary(summary, i)
                summary_writer.flush()

        if epoch % 1 == 0:

            # calculate the average loss and accuracy for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_accuracy = epoch_accuracy / num_batches

            logger.info('Epoch: {}, loss: {:.3f}'.format(epoch, avg_epoch_loss))
            logger.info('Train accuracy {:.3f}'.format(avg_epoch_accuracy))

        if epoch % 1 == 0:

            # show cross-validation accuracy
            # notice we don't drop-out when checking accuracy on the test set
            logger.info('CV accuracy: {:.3f}'.format(
                accuracy.eval(feed_dict={inputs: cv_X,
                                         labels: cv_y,
                                         in_training: False})))
    # flush the writer
    summary_writer.close()
