"""
Keras script for training a model from preprocessed voice data.

Usage:

with downloaded data:

python3 train_keras.py --path_to_data=/a_path_to_the_downloaded_data

OR with randomly generated data:

python3 train_keras.py --random_data=True
"""

import argparse
import bcolz
import numpy as np
import os
import time

# keras as tensorflow backend
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv1D
from tensorflow.python.keras.layers import Input, MaxPooling1D, GlobalMaxPool1D, Activation
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model

# config
DEFAULT_PATH_TO_DATA = os.path.join('data', 'main', 'preprocessed')
NUM_CATEGORIES = 12
NUM_EPOCHS = 2
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0001


# define the bcolz array saving functions
def bcolz_save(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()


def bcolz_load(file_name):
    return bcolz.open(file_name)[:]


def load_data(data_path):
    """
    Function for loading the X and y for the voice recognition challenge.
    Assumes previous steps from the jupyter notebooks were followed.
    :param data_path: a path to the preprocessed bcolz data sets
    """

    # load the y
    print('\nLoading the y ...')
    train_y = bcolz_load(data_path + os.path.sep + 'train_y' + '.bc')
    cv_y = bcolz_load(data_path + os.path.sep + 'cv_y' + '.bc')

    # report y shapes
    print('Shape of train y: {}'.format(train_y.shape))
    print('Shape of cv y: {}'.format(cv_y.shape))

    # load the X
    print('\nLoading the X ...')
    cv_X = bcolz_load(data_path + os.path.sep + 'cv_X' + '.bc')

    train_Xs = []
    for i in range(7):
        train_subset = bcolz_load(
            data_path + os.path.sep + 'train_X' + str(i + 1) + '.bc')
        train_Xs.append(train_subset)

    # split the train y
    train_ys = []
    subset_size = 3168
    for i in range(7):
        train_y_subset = train_y[subset_size * i: subset_size * (i + 1)]
        train_ys.append(train_y_subset)

    # expand the train X
    expanded_train_Xs = [np.expand_dims(train_X, axis=2) for train_X in
                         train_Xs]
    print('\nShape of train X, expanded:', expanded_train_Xs[0].shape)

    # expand the cv X
    expanded_cv_X = np.expand_dims(cv_X, axis=2)
    print('Shape of cv X, expanded:', expanded_cv_X.shape)

    return (expanded_train_Xs, train_ys,
            expanded_cv_X, cv_y,)


def generate_random_data():
    """
    Function for generating random X and y for the voice recognition challenge.
    :return: X and y for all the train and cv subsets
    """

    # generate the X
    print('\nGenerating the X sets ...')
    expand_train_Xs = []

    for j in range(7):
        expand_train_X = np.random.rand(3168, 16000, 1)
        expand_train_Xs.append(expand_train_X)

    expand_val_X = np.random.rand(3087, 16000, 1)

    # report shapes of X
    print('X train subset shape: {}'.format(expand_train_Xs[0].shape))
    print('X cv shape: {}'.format(expand_val_X.shape))

    # generate the y
    print('\nGenerating the y sets ...')
    training_ys = []

    for i in range(7):
        training_y = np.random.rand(3168, 12)
        training_ys.append(training_y)

    val_y = np.random.rand(3087, 12)

    # report shapes of y
    print('y train subset shape: {}'.format(training_ys[0].shape))
    print('y cv subset shape: {}'.format(val_y.shape))

    return (expand_train_Xs, training_ys,
            expand_val_X, val_y,)


def create_model():
    """
    Define and return a keras Model() instance.
    """
    # input layer & batch normalization
    inputs = Input(shape=(16000, 1))
    x_1d = BatchNormalization(name='batchnormal_1d_in')(inputs)

    # iteratively create 9 blocks of 2 convolutional
    # layers with batchnorm and max-pooling
    for i in range(9):
        name = 'step' + str(i)

        # first 1D convolutional block
        x_1d = Conv1D(8 * (2 ** i), (3), padding='same',
                      name='conv' + name + '_1')(x_1d)
        x_1d = BatchNormalization(name='batch' + name + '_1')(x_1d)
        x_1d = Activation('relu')(x_1d)

        # second 1D convolutional block
        x_1d = Conv1D(8 * (2 ** i), (3), padding='same',
                      name='conv' + name + '_2')(x_1d)
        x_1d = BatchNormalization(name='batch' + name + '_2')(x_1d)
        x_1d = Activation('relu')(x_1d)

        # max pooling
        x_1d = MaxPooling1D((2), padding='same')(x_1d)

    # final convolution and dense layer
    x_1d = Conv1D(1024, (1), name='last1024')(x_1d)
    x_1d = GlobalMaxPool1D()(x_1d)
    x_1d = Dense(1024, activation='relu', name='dense1024_onlygmax')(x_1d)
    x_1d = Dropout(DROPOUT_RATE)(x_1d)

    # soft-maxed prediction layer
    predictions = Dense(NUM_CATEGORIES, activation='softmax', name='cls_1d')(
        x_1d)

    a_model = Model(inputs=inputs, outputs=predictions)
    a_model.compile(Adam(lr=LEARNING_RATE), loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return a_model


def create_parser():
    """
    Create a simple command line argument parser for the data directory.
    :return: an argparse.ArgumentParser() instance
    """
    a_parser = argparse.ArgumentParser()

    # for data path
    a_parser.add_argument("--path_to_data",
                          type=str,
                          default=DEFAULT_PATH_TO_DATA,
                          help="Path to the data .bc files' parent directory")
    # else, for random data
    a_parser.add_argument("--random_data",
                          type=bool,
                          default=False,
                          help="Boolean marking whether random data is to be"
                               " used (when actual data has not been "
                               "donwloaded).")

    return a_parser


if __name__ == '__main__':

    # add parser
    parser = create_parser()
    params = parser.parse_args()

    # start
    print('\nStarting the keras training script...')

    # grab the user-specified path to data or the default
    path_to_data = params.path_to_data

    # grab the boolean whether to use downloaded or random-generated data
    use_random_data = params.random_data

    # handle data source
    if use_random_data:
        # generate random data
        data = generate_random_data()

    else:
        # load the downloaded data
        data = load_data(path_to_data)

    # unpack the data
    expanded_train_Xs, train_ys, expanded_cv_X, cv_y = data

    # create the model
    model = create_model()

    # time it
    start = time.time()

    # train
    for j in range(NUM_EPOCHS):

        # fit iteratively
        for i, expanded_train_X in enumerate(expanded_train_Xs):
            # pretty printing
            print(i + 1, '/', len(expanded_train_Xs))

            result = model.fit(expanded_train_X, train_ys[i],
                               batch_size=BATCH_SIZE,
                               epochs=1,
                               validation_data=(expanded_cv_X, cv_y))

            # pretty printing
            duration = time.time() - start
            print('Took {:.2f} seconds\n'.format(duration))

            # results
            cv_acc = '{:.4f}'.format(result.history['val_acc'][0]).replace('.', '')
            train_acc = '{:.4f}'.format(result.history['acc'][0]).replace('.', '')