"""The BMT dataset from Melania Pintilie's book "Competing Risks, A
Practical Perspective".

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata

ARCHIVE = RemoteFileMetadata(
    filename=None, url=('local:BMT_data'), checksum=None)

def fetch_BMT(data_home=None, download_if_missing=True,
              random_state=None, shuffle=False,
              categ_as_strings=False, return_X_y=False,
              as_frame=False):
    """Load the BMT (Bone Marrow Transplant) dataset from Melania
    Pintilie's book "Competing Risks, A Practical Perspective.

    Use a local copy of the data.

    The agvhdgd variable (Grade of acute GVHD) is treated as another
    target.

    **Targets**

    - target_surv_time: survival time
    - target_surv_status: survival censoring status  1=death
    - target_relapse_time: time to relapse
    - target_relapse_status: 1=relapse
    - target_agvh_time: time to AGVH
    - target_agvh: 1=AGVH
    - target_agvhdgd: AGVH grade 0 (absent) - 4, ordinal scale
    - target_cgvh_time: time to CGVH
    - target_cgvh: 1=CGVH


    
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

    dataset.DESCR : string
        Description of the dataset.

    (data, target_time, target_status) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['BM', 'PB']
    dx_values = ['CML', 'AML']
    extent_values = ['L', 'E'] 
    agvhdgd_values = ['0', '1', '2', '3', '4'] 

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "tx")]
    target_descr = [("target_surv_time", float, "survtime"),
                    ("target_surv_status", np.int32, "stat"),
                    ("target_relapse_time", float, "reltime"),
                    ("target_relapse_status", np.int32, "rcens"),
                    ("target_agvh_time", float, "agvhtime"),
                    ("target_agvh", np.int32, "agvh"),
                    ("target_agvhdgd", agvhdgd_values, "agvhdgd"),
                    ("target_cgvh_time", float, "cgvhtime"),
                    ("target_cgvh", np.int32, "cgvh"),
                    ]

    feature_descr = [("dx", dx_values),
                     ("extent", extent_values),
                     ("age", float),
    ]

    ret = _fetch_remote_csv(ARCHIVE, "BMT",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=13
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
    
