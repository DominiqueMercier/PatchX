import numpy as np
import pandas as pd
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import impute

from modules.data_processor import group_classwise


def create_clf_features(data, labels=None, pre_selected=None, mode='all', return_names=False):
    """Method to create a clf that is based on the extracted features.

    Args:
        data (arr): dataset array
        labels (arrr, optional): label array. Defaults to None.
        pre_selected (arr, optional): preselected features. Defaults to None.
        mode (str, optional): feature selection mode. can be given, relevant or all. Defaults to 'all'.
        return_names (bool, optional): flag to return the feature names used. Defaults to False.

    Returns:
        objects: either the features array or the features array and the names
    """
    data_ids = np.expand_dims(
        np.repeat(np.arange(data.shape[0]), data.shape[1]), axis=1)
    data_ts = np.expand_dims(
        np.hstack((np.arange(data.shape[1]),) * data.shape[0]), axis=1)
    data_trans = np.reshape(data, (-1, data.shape[2]))
    data_con = np.concatenate([data_ids, data_ts, data_trans], axis=1)

    col_names = ['ID', 'TS'] + ['C' + str(i) for i in range(data.shape[2])]
    pd_data = pd.DataFrame(data=data_con, columns=col_names)

    if mode == 'relevant':
        pd_labels = pd.Series(data=labels)
        features = extract_relevant_features(
            pd_data, pd_labels, column_id='ID', column_sort='TS')
    elif 'given':
        features = extract_features(
            pd_data, kind_to_fc_parameters=pre_selected, column_id='ID', column_sort='TS')
    else:
        features = extract_features(pd_data, column_id='ID', column_sort='TS')
    if not mode == 'relevant':
        features = impute(features)
    print('Number of extracted features:', len(list(features)))
    if return_names:
        feature_names = from_columns(features)
        return features, feature_names
    else:
        return features


def compute_relevant_subset_features(data, labels, num_classes, num=500):
    """Method to extract the relevant feature subset in case it is too large.

    Args:
        data (arr): dataset array
        labels (arr): label array
        num_classes (int): number of classes (sparse)
        num (int, optional): number of features to keep at max. Defaults to 500.

    Returns:
        arr: relevant features
    """
    cwise = group_classwise(labels, num_classes)
    per_class = num // num_classes
    cwise = np.array([c[i] for c in cwise for i in range(per_class)])
    part_data = data[cwise]
    part_labels = labels[cwise]
    _, relevant_features = create_clf_features(
        part_data, labels=part_labels, mode='relevant', return_names=True)
    return relevant_features
