"""The pbc datasets from R survival package.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="pbc.csv", url='local:pbc_data', checksum=None)

def _float_w_nan(x):
    """Convert strings to floats with empty strings converted to
    nan's."""
    y = [v if v != "" else "nan" for v in x]
    return np.array(y, float), float

def fetch_pbc(data_home=None, download_if_missing=True,
              random_state=None, shuffle=False,
              categ_as_strings=False, return_X_y=False,
              as_frame=False):
    """Load the pbc dataset from R survival package (uplift survival).

    Download it if necessary.

    Only first 312 records with assigned treatment are kept.
    
    Following the original dataset, the edema variable is numerical
        but can also be treated as categorical: 0 no edema, 0.5
        untreated or successfully treated, 1 edema despite diuretic
        therapy

    **Variables**

    chol, copper, trig, platelet contain missing data
    
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

    dataset.target_status : numpy array
        Censoring status: 0=censored, 1=transplant, 2=dead.

    dataset.target_time : numpy array
        Censoring, transplant or death time.

    dataset.DESCR : string
        Description of the dataset.

    (data, target_time, target_status) : tuple if
        ``return_X_y`` is True

    """

    target_status_values = {"0":"censored", "1":"transplant", "2":"dead"}
    treatment_values = {"2":"placebo", "1":"D-penicillamine"}
    sex_values = ["f", "m"]
    stage_values = ["1", "2", "3", "4"]

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "trt")]
    target_descr = [("target_status", target_status_values, "status"),
                    ("target_time", float, "time"),]

    feature_descr = [("age", float),
                     ("sex", sex_values),
                     ("ascites", np.int32),
                     ("hepato", np.int32),
                     ("spiders", np.int32),
                     ("edema", float),
                     ("bili", float),
                     ("chol", _float_w_nan),
                     ("albumin", float),
                     ("copper", _float_w_nan),
                     ("alk.phos", float),
                     ("ast", float),
                     ("trig", _float_w_nan),
                     ("platelet", _float_w_nan),
                     ("protime", float),
                     ("stage", stage_values),
    ]

    ret = _fetch_remote_csv(ARCHIVE, "pbc",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=19
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
    
