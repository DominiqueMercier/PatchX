import os
import pickle

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_ucr(path):
    """Loads train and test data from UCR TRAIN and TEST file.

    Args:
        path (str): path to the folder containing TRAIN and TEST

    Returns:
        objects: trainX, trainY, None, None, testX, testY
    """
    name = path.split(os.path.sep)[-2]
    train_path = os.path.join(path, name + '_TRAIN')
    test_path = os.path.join(path, name + '_TEST')

    trainX, trainY = read_data(train_path)
    testX, testY = read_data(test_path)
    return trainX, trainY, None, None, testX, testY


def read_data(path):
    """Reads the data and labels from file and converts the labels to suitable sparse labels.

    Args:
        path (str): path to the text file

    Returns:
        objects: data, labels. both are arrays
    """
    data = []
    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            com_sep = line.split(',')
            data.append(com_sep[1:])
            labels.append(com_sep[0])
    data = np.expand_dims(np.array(data, dtype=float), axis=-1)
    labels = np.array(labels)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    return data, labels


def load_data(path):
    """Loads a pickle file. Takes care of the existence of a validation dataset and the channel format.

    Args:
        path (str): path to the pickle file containing all data and labels

    Returns:
        objects: trainX, trainY, valX, valY, testX, testY. valX and valY may be None
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    valX, valY = None, None
    if len(d) < 6:
        trainX, trainY, testX, testY = d
    else:
        trainX, trainY, valX, valY, testX, testY = d
    # create channel axis
    if len(trainX.shape) < 3:
        trainX = np.expand_dims(trainX, axis=-1)
        if valX is not None:
            valX = np.expand_dims(valX, axis=-1)
        testX = np.expand_dims(testX, axis=-1)
    # create sparse labels
    if len(trainY.shape) > 1:
        trainY = np.argmax(trainY, axis=-1)
        if valY is not None:
            valY = np.argmax(valY, axis=-1)
        testY = np.argmax(testY, axis=-1)
    else:
        trainY = np.array(trainY, dtype=int)
        if valY is not None:
            valY = np.array(valY, dtype=int)
        testY = np.array(testY, dtype=int)
    return trainX, trainY, valX, valY, testX, testY


def get_shapes(data, labels):
    """Returns the dataset statistics such as sequence length, channel, and classes.

    Args:
        data (arr): dataset
        labels (arr): sparse labels

    Returns:
        objects: channel number, sequence length, number of classes
    """
    seqlen = data.shape[1]
    channel = data.shape[2]
    classes = len(np.unique(labels))
    return channel, seqlen, classes


def split_dataset(data, labels, split=0.3, stratify=None):
    """Splits the data into two sets

    Args:
        data (arr): dataset
        labels (arr): labels in a sparse format
        split (float, optional): split factor for the test size. Defaults to 0.3.
        stratify (arr, optional): Labels used for stratified split. Defaults to None.

    Returns:
        objects: X1, X2, Y1, Y2, all are arrays
    """
    X1, Y1, X2, Y2 = train_test_split(
        data, labels, test_size=split, random_state=1, stratify=stratify)
    return X1, X2, Y1, Y2


def group_classwise(labels, num_classes):
    """Method to group the ids labelwise

    Args:
        labels (arr): sparse label array
        num_classes (int): number of classes

    Returns:
        list: list with ids for each label
    """
    cwise = [[] for i in range(num_classes)]
    for i in range(len(labels)):
        cwise[labels[i]].append(i)
    cwise = [np.array(c) for c in cwise]
    return cwise


def generate_data(path, create_val=True, useUCR=False, verbose=1):
    """Loads the data using pickle files and create the required datasplits.

    Args:
        path (str): path to the pickle files in case of UCR to the directory containing the TRAIN and TEST file.
        create_val (bool, optional): Flag to create a validation out of th training set. Defaults to True.
        verbose (int, optional): Flag to verbose the processing and data statistics. Defaults to 1.

    Returns:
        objects: trianX, trainY, valX, valY, testX, testY, number of classes, sequence length and channels
    """
    if useUCR:
        trainX, trainY, valX, valY, testX, testY = load_ucr(path)
    else:
        trainX, trainY, valX, valY, testX, testY = load_data(path)
    if create_val and valX is None:
        trainX, trainY, valX, valY = split_dataset(
            trainX, trainY, stratify=trainY)

    channel, seqlen, classes = get_shapes(trainX, trainY)
    if verbose:
        print('Finished data preprocessing')
        print('='*40)
        print('Dataset statistics')
        print(f'Train: {trainX.shape}')
        if valX is not None:
            print(f'Val: {valX.shape}')
        print(f'Test: {testX.shape}')
        print(f'Classes: {classes}')
        print(f'Sequence length: {seqlen}')
        print(f'Channel: {channel}')
        print('='*40)
    return trainX, trainY, valX, valY, testX, testY, classes, seqlen, channel
