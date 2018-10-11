import tensorflow as tf
import bcolz
import numpy as np
import os

# constants
NUM_SUBSETS = 7
SUBSET_SIZE = 3618

# paths
path_to_main = os.path.join("data", "main")
path_to_main_preprocessed = os.path.join(path_to_main, "preprocessed")

# define the bcolz array saving functions
def bcolz_save(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def bcolz_load(fname): return bcolz.open(fname)[:]

# load the y
print("Loading the y sets ...")
train_y = bcolz_load(path_to_main_preprocessed + os.path.sep + "train_y" + ".bc")
cv_y = bcolz_load(path_to_main_preprocessed + os.path.sep + "cv_y" + ".bc")
test_y = bcolz_load(path_to_main_preprocessed + os.path.sep + "test_y" + ".bc")

# reload the Test & CV X
print("Loading the X sets ...")
cv_X = bcolz_load(path_to_main_preprocessed + os.path.sep + "cv_X" + ".bc")
test_X = bcolz_load(path_to_main_preprocessed + os.path.sep + "test_X" + ".bc")

# reload the Train X
train_Xs = []
for i in range(NUM_SUBSETS):
    train_subset = bcolz_load(path_to_main_preprocessed + os.path.sep + "train_X" + str(i + 1) +".bc")
    train_Xs.append(train_subset)
    print("Loading the X train subset #{} ...".format(i + 1))
    print("X subset shape: {}".format(train_subset.shape))

# train X subsets have 3168 examples each (7 total), exactly
train_ys = []
for i in range(NUM_SUBSETS):
    print("Splitting the y train set into subset #{}...".format(i + 1))
    train_y_subset = train_y[SUBSET_SIZE * i: SUBSET_SIZE * (i + 1)]
    print("Y subset shape: {}".format(train_y_subset.shape))
    train_ys.append(train_y_subset)
