"""The Starbucks dataset.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="training.csv",
    url=('https://raw.githubusercontent.com/01KAT1/'
         'Marketing-Promotion-Campaign-Uplift-Modelling-Starbucks-Dataset/'
         'main/training.csv'),
    checksum=('2ac9d5c601b134b9b69e742b97d01652'
              '9c8082f6a611d5b243ab0c5b2ceaf83e'))

def fetch_Starbucks(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False,
                    as_frame=False):
    """Load the Starbucks dataset.

    Download it if necessary.  There are many versions of this
    dataset, here the one from
    https://raw.githubusercontent.com/01KAT1/Marketing-Promotion-Campaign-Uplift-Modelling-Starbucks-Dataset/main/training.csv
    is used since it is easy to use and has been used in many uplift
    modeling papers.  An original version consisting of several tables
    can be found at https://github.com/Shuniy/starbucks/tree/main

    See also an online post about analyzing the data:
    https://medium.com/@nesreensada/how-to-build-a-profitable-promotion-strategy-easily-with-uplift-modeling-26b2addc3e46
    
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

    dataset.target_purchase : numpy array
        Indicator whether a purchase was made.

    dataset.DESCR : string
        Description of the dataset.

    (data, target_purchase) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['No', 'Yes']
    V1_values = ["0", "1", "2", "3"]
    V4_values = ["1", "2"]
    V5_values = ["1", "2", "3", "4"]
    V6_values = ["1", "2", "3", "4"]
    V7_values = ["1", "2"]

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "Promotion")]
    target_descr = [("target_purchase", np.int32, "purchase"),]

    feature_descr = [#("id", np.int32),
                     ("V1", V1_values),
                     ("V2", float),
                     ("V3", float),
                     ("V4", V4_values),
                     ("V5", V5_values),
                     ("V6", V6_values),
                     ("V7", V7_values),
    ]

    ret = _fetch_remote_csv(ARCHIVE, "Starbucks",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=10
                            )
    if not return_X_y:
        ret.descr = __doc__

    return ret
