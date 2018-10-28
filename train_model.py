import tensorflow as tf
import bcolz
import numpy as np
import os

# constants
NUM_X_SUBSETS = 7
X_SIZE = 16000
NUM_CLASSES = 12
LEARNING_RATE = 0.0001

# log level
tf.logging.set_verbosity(tf.logging.DEBUG)

# paths
path_to_main = os.path.join("data", "main")
path_to_main_preprocessed = os.path.join(path_to_main, "preprocessed")


# define the bcolz array saving functions
def bcolz_save(fname, arr): c = bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def bcolz_load(fname): return bcolz.open(fname)[:]


if __name__ == "__main__":

    # load the y
    print("Loading the y sets ...")
    train_y = bcolz_load(path_to_main_preprocessed + os.path.sep + "train_y" + ".bc")
    cv_y = bcolz_load(path_to_main_preprocessed + os.path.sep + "cv_y" + ".bc")
    test_y = bcolz_load(path_to_main_preprocessed + os.path.sep + "test_y" + ".bc")

    print("y train shape: {}".format(train_y.shape))
    print("y cv shape: {}".format(cv_y.shape))
    print("y test shape: {}\n".format(test_y.shape))

    # reload the Test & CV X
    print("Loading the X sets ...")
    cv_X = bcolz_load(path_to_main_preprocessed + os.path.sep + "cv_X" + ".bc")
    test_X = bcolz_load(path_to_main_preprocessed + os.path.sep + "test_X" + ".bc")

    # reload the Train X
    print("Loading the X train subsets ...")
    train_Xs = []
    for i in range(NUM_X_SUBSETS):
        train_subset = bcolz_load(path_to_main_preprocessed + os.path.sep + "train_X" + str(i + 1) + ".bc")
        # print("X subset shape: {}".format(train_subset.shape))
        train_Xs.append(train_subset)

    # concatenate the Train Xs back into one matrix
    print("Concatenating the X train subsets...")
    train_X = np.concatenate(tuple(train_Xs))

    print("X train shape: {}".format(train_X.shape))
    print("X cv shape: {}".format(cv_X.shape))
    print("X test shape: {}\n".format(test_X.shape))

    # we need to expand the dimensions for 1D convolutions
    print("Expanding the dimensions of the X train set for convolutional layers...")
    train_X = np.expand_dims(train_X, axis=2)
    print("Expanded X train shape: {}".format(train_X.shape))

    # same for CV & test
    print("Expanding the dimensions of the X cv & test subsets ...")
    cv_X = np.expand_dims(cv_X, axis=2)
    print("Expanded X cv shape: {}".format(cv_X.shape))
    test_X = np.expand_dims(test_X, axis=2)
    print("Expanded X test shape: {}".format(test_X.shape))

    # TODO: train a simple 1D CNN
    # input is too big to be a variable, has to be a placeholder that is then fed via a feed dict
    X = tf.placeholder(tf.float32, shape=(train_X.shape[0], X_SIZE, 1))
    y = tf.placeholder(tf.float32, shape=(train_y.shape[0], NUM_CLASSES))

    # declare a variable-initializing op
    init = tf.global_variables_initializer()

    # declare an op to control dropout
    keep_prob = tf.placeholder(tf.float32)

    # tf session
    with tf.Session() as sess:

        # MODEL ARCHITECTURE
        x = tf.layers.batch_normalization(inputs=X)
        x = tf.layers.conv1d(x, filters=64, kernel_size=3, padding='same')
        x = tf.nn.relu(x)
        # maxpool followed by flatten = global maxpool
        x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding="same")
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=1000)
        x = tf.layers.dense(x, units=NUM_CLASSES)
        preds = tf.nn.softmax(x)

        # MODEL TRAINING
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # run the session, feeding it the appropriate data
        sess.run(init, feed_dict={X: train_X, y: train_y})
