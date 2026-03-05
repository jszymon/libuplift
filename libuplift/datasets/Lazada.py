"""Lazada E-comerce dataset.

See https://github.com/kailiang-zhong/DESCN/tree/main/data/Lazada_dataset/ for details.
"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE_TRAIN = RemoteFileMetadata(
    filename="Lazada_train.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Lazada/Lazada_trainset.csv.gz'),
    checksum=('5a46ec368f9e8397267e818447ea7f3d'
              '239a671731ca98c0ae9f17c19ab5a469'))
ARCHIVE_TEST = RemoteFileMetadata(
    filename="Lazada_test.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Lazada/Lazada_testset.csv.gz'),
    checksum=('a6d8832a71e8f0b6d4e8858d91ce7de9'
              'd8e0f2f58aa7208c78b1e0f7d6ba9c94'))


def fetch_Lazada(version="test", data_home=None, download_if_missing=True,
                 random_state=None, shuffle=False,
                 categ_as_strings=False, return_X_y=False,
                 as_frame=False):
    """Load the Lazada e-comerce dataset.

    Download it if necessary.

    There is a training and test set available.  The test dataset
    comes from a randomized experiment, in the training set assignment
    is biased.
    
    Details of the dataset can be found in the following paper:
    https://arxiv.org/pdf/2207.09920.

    The license is available `here <https://github.com/jszymon/uplift_sklearn_data/releases/download/Lazada/Lazada_LICENSE>`__

    Parameters
    ----------
    version : string, optional
        Specify whether to return training ('train') or testing
        ('test') dataset.  Test dataset comes from a randomized
        experiment, in the training set assignment is biased.

    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    categ_as_strings : bool, default=False
        Whether to return categorical variables as strings.

    return_X_y : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    as_frame : boolean, default=False
        If True features are returned as pandas DataFrame.  If False
        features are returned as object or float array.  Float array
        is returned if all features are floats.
    
    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array
        Each row corresponds to the features in the dataset.

    dataset.target_visit : numpy array
        Each value is 1 if website visit occurred 0 otherwise.

    dataset.target_conversion : numpy array
        Each value is 1 if purchase occurred 0 otherwise.

    dataset.target_spend : numpy array
        Each value corresponds to the amount of money spent.

    dataset.DESCR : string
        Description of the Hillstrom dataset.

    (data, target_conversion, target_visit, target_exposure) : tuple if
        ``return_X_y`` is True

    """

    if version == "train":
        ARCHIVE = ARCHIVE_TRAIN
        file_name = "Lazada_train"
    elif version == "test":
        ARCHIVE = ARCHIVE_TEST
        file_name = "Lazada_test"
    else:
        raise ValueError(f"Wrong version ({version}) of Lazada dataset"
                         "requested.\nValid choices are `train' or `test'")

    # dictionaries
    treatment_values = ['control', 'treated']
    categ_values = dict()

    # attribute descriptions
    treatment_descr = [("treatment", np.int32, "is_treat")]
    target_descr = [("target", np.int32, "label")]
    feature_descr = [(f"f{i}", float) for i in range(83)]

    ret = _fetch_remote_csv(ARCHIVE, file_name,
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=85, all_num=True
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
