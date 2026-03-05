"""The Megafon dataset.

The version used comes from the scikit-uplift package.
"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="megafon_dataset.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Megafon/megafon_dataset.csv.gz'),
    checksum=('cdcb2d052b90f8eefa75937d5540f114'
              'd0748ea231d95e2778dd6760478e4a00'))

def fetch_Megafon(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False,
                    as_frame=False):
    """Load the Megafon dataset.

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

    dataset.target_conversion : numpy array
        Indicator whether a conversion occurred.

    dataset.DESCR : string
        Description of the dataset.

    (data, target_conversion) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['control', 'treatment']

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "treatment_group")]
    target_descr = [("target_conversion", np.int32, "conversion"),]

    feature_descr = [("X_"+str(i+1), float) for i in range(50)]

    ret = _fetch_remote_csv(ARCHIVE, "Megafon",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=52
                            )
    if not return_X_y:
        ret.descr = __doc__

    return ret
