import tensorflow as tf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def create_model(input_shape, num_classes, verbose=1):
    """Creates a 1d convolutional model.

    Args:
        input_shape (arr): input shape including number of time-steps and channels
        num_classes (int): number of classes
        verbose (int, optional): flag to return the summary of the model. Defaults to 1.

    Returns:
        object: model object
    """
    inputs = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1],))

    x = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x = tf.keras.layers.Dense(num_classes)(x)
    x = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.models.Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()

    return model


def create_dense_model(input_shape, num_classes, verbose=1):
    """Creates a 1d dense model.

    Args:
        input_shape (arr): input shape including number of time-steps and channels
        num_classes (int): number of classes
        verbose (int, optional): flag to return the summary of the model. Defaults to 1.

    Returns:
        object: model object
    """
    inputs = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1],))

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    x = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.models.Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()

    return model


def create_clf(clf_type, balanced=False):
    """Creates a traditional machine learning approach.

    Args:
        clf_type (str): either svm or random forest
        balanced (bool, optional): flag to use balanced class weights. Defaults to False.

    Returns:
        object: classifier object
    """
    if clf_type == 'svm':
        return create_svm(balanced=balanced)
    else:
        return create_random_forest(balanced=balanced)


def create_svm(linear=False, balanced=False):
    """Creates an SVM

    Args:
        linear (bool, optional): flag to use linear svm. Defaults to False.
        balanced (bool, optional): flag to balance the classes. Defaults to False.

    Returns:
        object: svm classifier
    """
    kernel = 'rbf'
    class_weight = None
    if linear:
        kernel = 'linear'
    if balanced:
        class_weight = 'balanced'

    clf = svm.SVC(kernel=kernel, class_weight=class_weight)
    return clf


def create_random_forest(balanced=False):
    """Creates a random forest

    Args:
        balanced (bool, optional): flag to blaance the classes. Defaults to False.

    Returns:
        object: random forest classifier
    """
    class_weights = None
    if balanced:
        class_weights = 'balanced'
    clf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, class_weight=class_weights)
    return clf
