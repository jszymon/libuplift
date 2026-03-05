"""The colon datasets from R survival package.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="colon.csv",
    url=('https://vincentarelbundock.github.io/'
         'Rdatasets/csv/survival/colon.csv'),
    checksum=('6f3472a64f696e3195daa198f054180c'
              '3e4c66408f7fb8c548c6f4c7b8f898ee'))

def _float_w_nan(x):
    """Convert strings to floats with empty strings converted to
    nan's."""
    y = [v if v != "" else "nan" for v in x]
    return np.array(y, float), float

def fetch_colon(data_home=None, download_if_missing=True,
                random_state=None, shuffle=False,
                categ_as_strings=False, return_X_y=False,
                as_frame=False):
    """Load the colon dataset from R survival package (uplift survival).

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

    dataset.target_recurrence_time : numpy array
        Survival or censoring time.

    dataset.target_recurrence_status : numpy array
        Censoring. Each value is 0 (censored) or 1 (event).

    dataset.target_death_time : numpy array
        Survival or censoring time.

    dataset.target_death_status : numpy array
        Censoring. Each value is 0 (censored) or 1 (event).

    dataset.DESCR : string
        Description of the dataset.

    (data, target_time, target_status) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['Obs', 'Lev', 'Lev+5FU']
    differ_values = {"1":"well", "2":"moderate", "3":"poor", "":"NA"}     
    extent_values = {"1":"submucosa", "2":"muscle", "3":"serosa", 
                     "4":"contiguous_structures"} 

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "rx")]
    target_descr = [("target_recurrence_time", float, "time"),
                    ("target_recurrence_status", np.int32, "status"),]

    feature_descr = [#("rownames", np.int32),
                     #("id", np.int32),
                     #("study", np.int32),
                     ("sex", np.int32),
                     ("age", float),
                     ("obstruct", np.int32),
                     ("perfor", np.int32),
                     ("adhere", np.int32),
                     ("nodes", _float_w_nan),
                     ("differ", differ_values),
                     ("extent", extent_values),
                     ("surg", np.int32),
                     ("node4", np.int32),
                     ("etype", np.int32),
    ]

    ret = _fetch_remote_csv(ARCHIVE, "colon",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=17
                            )
    if not return_X_y:
        ret.descr = __doc__

    # extract different targets
    ret.target_names = ["target_recurrence_time", "target_recurrence_status",
                        "target_death_time", "target_death_status",]
    ret.feature_names.remove("etype")
    if as_frame:
        etype = ret.data["etype"]
        ret.data = ret.data[etype==1].reset_index().drop("etype", axis=1)
    else:
        etype = ret.data[:, -1]
        ret.data = ret.data[etype==1,:-1]
    ret.treatment = ret.treatment[etype==1]
    ret.target_death_time = ret.target_recurrence_time[etype==2]
    ret.target_death_status = ret.target_recurrence_status[etype==2]
    ret.target_recurrence_time = ret.target_recurrence_time[etype==1]
    ret.target_recurrence_status = ret.target_recurrence_status[etype==1]
    return ret
