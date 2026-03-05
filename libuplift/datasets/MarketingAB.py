"""An A/B testing dataset from Kaggle.
"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="marketing_AB.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/releases/download/marketing_AB/marketing_AB.csv.gz'),
    checksum=('a318767a73785b7e54fbf36cbf300411'
              '5f61e943844b7ac8d3753a893e093d9d'))

def fetch_marketing_AB(data_home=None, download_if_missing=True,
                       random_state=None, shuffle=False,
                       categ_as_strings=False, return_X_y=False,
                       as_frame=False):
    """Load the marketing_AB dataset from Kaggle.
    Download it if necessary.
    
    See https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing
    for details.

    The treatment was showing the user an advertisement ('ad'), the
    control showing a Public Service Announcement ('psa').

    The dataset exhibits very high class and treatment imbalance.

    Changes made to the original dataset:
     * removed record number column
     * changed spaces to _ in column names
     * changed target from bool to {0,1}


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
    treatment_values = ['psa', 'ad']
    most_ads_day_values = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "test_group")]
    target_descr = [("target_converted", np.int32, "converted"),]

    feature_descr = [("user_id", float),
                     ("total_ads", float),
                     ("most_ads_day", most_ads_day_values),
                     ("most_ads_hour", np.int32),
    ]

    ret = _fetch_remote_csv(ARCHIVE, "marketing_AB",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=6,
                            )
    if not return_X_y:
        ret.descr = __doc__

    return ret
