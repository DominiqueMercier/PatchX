import numpy as np
from sklearn.metrics import classification_report


def get_classification_report(labels, preds, verbose=False):
    """Create a classification report.

    Args:
        labels (arr): array with the ground truth labels (sparse)
        preds (arr): sparse preditcions
        verbose (bool, optional): flag to print the report. Defaults to False.

    Returns:
        dict: dictionary of the classifiation report
    """
    if verbose:
        print(classification_report(labels, preds))
    return classification_report(labels, preds, output_dict=True)


def get_complete_evaluation(trainY, valY, testY, train_pred, val_pred, test_pred):
    """Perform accuracy computation for all datasets including training, validation, test.

    Args:
        trainY (arr): training labels
        valY (arr): validation labels
        testY (arr): test labels
        train_pred (arr): training predictions
        val_pred (arr): vlaidation predictions
        test_pred (arr): test predictions

    Returns:
        str: string holding the complete evaluation over all three datasets
    """
    train_rep = classification_report(trainY, train_pred, output_dict=True)
    val_rep = classification_report(valY, val_pred, output_dict=True)
    test_rep = classification_report(testY, test_pred, output_dict=True)
    s = 'Train Acc: %.4f | Val Acc: %.4f  | Test Acc: %.4f' % (
        train_rep['accuracy'], val_rep['accuracy'], test_rep['accuracy'])
    print(s)
    return s


def get_misclassifications(labels, preds):
    """Returns the misclassifierd ids.

    Args:
        labels (arr): array with the ground truth labels (sparse)
        preds (arr): sparse preditcions

    Returns:
        arr: array holding the ids of misclassified data
    """
    ids = []
    for i in range(labels.shape[0]):
        if labels[i] != preds[i]:
            ids.append(i)
    ids = np.array(ids)
    return ids


def compute_class_mean(data, labels):
    """Compute the class means

    Args:
        data (arr): dataset array
        labels (arr): label array (sparse)

    Returns:
        arr: returns the means of the classes
    """
    u, c = np.unique(labels, return_counts=True)
    class_means = np.zeros((len(u), data.shape[1]))
    for i in range(data.shape[0]):
        class_means[labels[i]] += data[i]
    for i in range(c.shape[0]):
        class_means[i] /= c[i]
    return class_means
