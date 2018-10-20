import tensorflow as tf
import bcolz
import numpy as np
import os

# constants
NUM_X_SUBSETS = 7
X_SIZE = 16000
NUM_CLASSES = 12

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
    expanded_cv_X = np.expand_dims(cv_X, axis=2)
    expanded_test_X = np.expand_dims(test_X, axis=2)

    # TODO follow this guy: https://github.com/easy-tensorflow/easy-tensorflow/blob/master/6_Convolutional_Neural_Network/code/main.py
    # NO - BUILD IT STEP BY STEP YOURSELF, STARTING WITH A SIMPLE MODEL
    # MAY HAVE TO JUST FOLLOW THE TF BOOK - FIND A 1D Conv example in it
