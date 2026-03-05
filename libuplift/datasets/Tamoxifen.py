"""The Tamoxifen dataset from Melania Pintilie's book "Competing
Risks, A Practical Perspective".

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata

ARCHIVE = RemoteFileMetadata(
    filename=None, url=('local:Tamoxifen_data'), checksum=None)

def fetch_Tamoxifen(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False,
                    as_frame=False):
    """Load the Tamoxifen randomized trial dataset from Melania
    Pintilie's book "Competing Risks, A Practical Perspective.

    The description of the original study can be found in [1]_.

    Uses a local copy of the data.
    
    **Targets**

    - target_surv_time: survival time
    - target_surv_status: 1=death
    - target_loctime: 
    - target_lcens: 1=local relapse
    - target_axltime: time to axillary relapse
    - target_acens: 1=axillary relapse
    - target_distime: time to distance relapse
    - target_dcens: 1=distance relapse
    - target_maltime: time to second malignancy
    - target_mcens: 1=second malignancy

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

    References
    ----------

    .. [1] A.W. Fyles, et al., "Tamoxifen with or without breast
       irradiation in women 50 years of age or older with early breast
       cancer". New England Journal of Medicine, 351(10), 963--970,
       2004 (https://www.nejm.org/doi/10.1056/NEJMoa040595).

    """

    # dictionaries
    treatment_values = {"T":"tamoxifen", "B":"radiation+tamoxifen"}
    hist_values = ["DUC", "LOB", "MED", "MIX", "MUC", "OTH"]
    hrlevel_values = ['NEG', 'POS']
    nodediss_values = ["N", "Y"]

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "tx")]
    target_descr = [("target_surv_time", float, "survtime"),
                    ("target_surv_status", np.int32, "stat"),
                    ("target_loctime", float, "loctime"),
                    ("target_lcens", np.int32, "lcens"),
                    ("target_axltime", float, "axltime"),
                    ("target_acens", np.int32, "acens"),
                    ("target_distime", float, "distime"),
                    ("target_dcens", np.int32, "dcens"),
                    ("target_maltime", float, "maltime"),
                    ("target_mcens", np.int32, "mcens"),
                    ]
    feature_descr = [("pathsize", float),
                     ("hist", hist_values),
                     ("hgb", float),
                     ("hrlevel", hrlevel_values),
                     ("nodediss", nodediss_values),
                     ("age", float),
                     ]

    ret = _fetch_remote_csv(ARCHIVE, "Tamoxifen",
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
    return ret
