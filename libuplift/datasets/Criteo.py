"""Criteo online advertising dataset.

See https://ailab.criteo.com/criteo-uplift-prediction-dataset/ for details.
"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


# original source
#ARCHIVE = RemoteFileMetadata(
#    filename="criteo-research-uplift-v2.1.csv.gz",
#    url=('https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz'),
#    checksum=('2716e1bf0fd157a93b5bf86924d90884'
#              '19dfbac2022c6cd90030220634f616dc'))
# GitHub copy, seems faster
ARCHIVE = RemoteFileMetadata(
    filename="criteo-research-uplift-v2.1.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/releases/download/Criteo/criteo-research-uplift-v2.1.csv.gz'),
    checksum=('2716e1bf0fd157a93b5bf86924d90884'
              '19dfbac2022c6cd90030220634f616dc'))


def fetch_Criteo(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False,
                    as_frame=False):
    """Load the Criteo dataset.

    Download it if necessary.

    Parameters
    ----------
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
        Description of the dataset.

    (data, target_conversion, target_visit, target_exposure) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['control', 'treated']
    categ_values = dict()

    # attribute descriptions
    treatment_descr = [("treatment", np.int32)]
    target_descr = [("target_conversion", np.int32, "conversion"),
                    ("target_visit", np.int32, "visit"),
                    ("target_exposure", np.int32, "exposure")]
    feature_descr = [(f"f{i}", float) for i in range(12)]

    ret = _fetch_remote_csv(ARCHIVE, "Criteo",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=16, all_num=True
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
