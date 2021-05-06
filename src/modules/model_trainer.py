import numpy as np
import tensorflow as tf
from joblib import dump

from modules.data_generator import DataGenerator


def train_blackbox(model_path, model, trainGen, valGen, epochs, verbose=1, workers=4):
    """Trains a blackbox model and includes callbacks for better performance.

    Args:
        model_path (str): path to save the model
        model (object): blackbox model
        trainGen (object): training data generator
        valGen (object): validation data generator
        epochs (int): number of epochs
        verbose (int, optional): flag to print progress. Defaults to 1.
        workers (int, optional): number of workers. Defaults to 4.

    Returns:
        [type]: [description]
    """
    trainGen.switch_shuffle(True)
    valGen.switch_shuffle(True)
    earlyStop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=6, verbose=verbose, mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=4, verbose=verbose, mode='auto', cooldown=0, min_lr=0)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, save_weights_only=False, monitor='val_loss')
    callback_list = [earlyStop, lr_reducer, mcp_save]

    model.fit(trainGen, validation_data=valGen, use_multiprocessing=True if workers > 1 else False,
              workers=workers, epochs=epochs, verbose=verbose, callbacks=callback_list)
    trainGen.switch_shuffle(False)
    valGen.switch_shuffle(False)
    model = tf.keras.models.load_model(model_path)
    return model


def train_descriptive(model_path, model, train_ids, val_ids, trainX, trainY, valX, valY, params, thresh=0.1, verbose=1, workers=4):
    """Method to train the explainable level 1 model. This method includes callbacks to get the best possible performance.

    Args:
        model_path (str): pathc to save the model
        model (object): model to train
        train_ids (arr): ids used to train the model
        val_ids (arr): ids used to validate the model
        trainX (arr): training dataset
        trainY (arr): training labels
        valX (arr): validation dataset
        valY (arr): validation labels
        params (dict): parameter dict for data generator
        thresh (float, optional): thresh to discard softmax values below. Defaults to 0.1.
        verbose (int, optional): flag to print training status. Defaults to 1.
        workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        object: trained model
    """
    train_part, val_part = np.copy(train_ids), np.copy(val_ids)
    tGen = DataGenerator(train_ids, trainX, trainY, **params)
    vGen = DataGenerator(val_ids, valX, valY, **params)
    tGen.switch_shuffle(True)
    vGen.switch_shuffle(True)

    best_acc = curr_acc = 0
    early_count = 0
    early_max = 1

    earlyStop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=6, verbose=verbose, mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=4, verbose=verbose, mode='auto', cooldown=0, min_lr=0)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, save_weights_only=False, monitor='val_loss')
    callback_list = [earlyStop, lr_reducer, mcp_save]

    while early_count < early_max:
        if verbose:
            print(f'Training on {train_part.shape[0]} / {train_ids.shape[0]}')
            print(f'Validate on {val_part.shape[0]} / {val_ids.shape[0]}')

        model.fit_generator(generator=tGen, validation_data=vGen, use_multiprocessing=True if workers > 1 else False,
                            workers=workers, epochs=2, verbose=verbose, callbacks=callback_list)
        curr_acc = model.evaluate(vGen, verbose=0)[1]
        if curr_acc >= best_acc:
            best_acc = curr_acc
            early_count = 0
        else:
            early_count += 1

        if verbose:
            print(f'Accuracy: {curr_acc} | Best Accuracy: {best_acc}')

        # select meaningful data
        tGen.switch_shuffle(False)
        vGen.switch_shuffle(False)
        train_pred = model.predict(tGen)
        val_pred = model.predict(vGen)

        train_part = select_meaningful(
            train_part, train_pred, thresh)
        tGen = DataGenerator(train_part, trainX, trainY, **params)
        val_part = select_meaningful(
            val_part, val_pred, thresh)
        vGen = DataGenerator(val_part, valX, valY, **params)
        tGen.switch_shuffle(True)
        vGen.switch_shuffle(True)

    if verbose:
        print('Optimized')

    model = tf.keras.models.load_model(model_path)
    return model


def select_meaningful(ids, preds, thresh):
    """Selects the data above the thresh

    Args:
        ids (arr): current ids used in set
        preds (arr): prediction values
        thresh (float): minimum thresh required to keep the sample

    Returns:
        arr: id array including only a subset of data
    """
    pred_max = np.max(preds, axis=-1)
    keep = np.squeeze(np.argwhere(pred_max > thresh))
    part = ids[keep]
    return part


def train_clf(model_path, clf, trainX, trainY, valX=None, valY=None):
    """Method to train the clf.

    Args:
        model_path (str): path to save the classifier
        clf (object): classifier object
        trainX (arr): training data arrray
        trainY (arr): sparse training labels
        valX (arr, optional): validation data array. Defaults to None.
        valY (arr, optional): sparse validation labels. Defaults to None.

    Returns:
        object: trained classifier
    """
    data, labels = trainX, trainY
    if not valX is None:
        data, labels = np.concatenate(
            [trainX, valX], axis=0), np.concatenate([trainY, valY], axis=0)
    if len(data.shape) > 2:
        data = np.reshape(data, (data.shape[0], -1))
    clf.fit(data, labels)
    dump(clf, model_path)
    return clf


def predict_clf(clf, data):
    """Method to perform clf prediction.

    Args:
        clf (object): classifier object
        data (arr): dataset used to perform prediction

    Returns:
        arr: prediction of clf
    """
    if len(data.shape) > 2:
        data = np.reshape(data, (data.shape[0], -1))
    pred = clf.predict(data)
    return pred
