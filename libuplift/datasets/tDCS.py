"""tDCS dataset from Kaggle.

A randomized controlled trial dataset for tDCS (transcranial direct current stimulation)
for allergic rhinitis treatment. Dataset from:
https://www.kaggle.com/datasets/ziya07/randomized-controlled-trial-dataset

The dataset contains information about patients undergoing tDCS treatment
for allergic rhinitis, including demographic information, treatment details,
and outcomes.
"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename=None,
    url="local:RCT_tDCS_Allergic_Rhinitist_csv",
    checksum=None)


def fetch_tDCS(data_home=None, download_if_missing=True,
              random_state=None, shuffle=False,
              categ_as_strings=False, return_X_y=False,
              as_frame=False):
    """Load the tDCS dataset (uplift classification).

    Download it if necessary.

    The treatment can be Sham (control) of tDCS.  There is also a
    polarity attribute describing treatment polarity '-' for controls.

    The main target variable is whether a petient positively responded
    to treatment.  Additional targets include measured biomarkers and
    symptom scores assessed at various time points after therapy.
    
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

    dataset.target : numpy array
        Each value is 1 if treatment was successful, 0 otherwise.

    dataset.treatment : numpy array
        Each value indicates the treatment group (0 or 1).

    dataset.DESCR : string
        Description of the tDCS dataset.

    (data, target, treatment) : tuple if ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ["Sham", "tDCS"]
    gender_values = ["F", "M"]
    polarity_values = ["-", "Anodal", "Cathodal"]

    
    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "Group"),
                       # TODO: continuous treatment descriptors:
                       #("current_mA", np.float64, "Current_mA"),
                       #("duration_min", np.float64, "Duration_min"),
                       ("polarity", polarity_values, "Polarity"),
                       ]
    target_descr = [("target", np.int32, "Response"),
                    ("post_IL6", np.float64, "Post_IL6"),
                    ("post_TNF_Alpha", np.float64, "Post_TNF_Alpha"),
                    ("post_IgE", np.float64, "Post_IgE"),
                    ("post_tDCS_symptom_score", np.float64, "Post_tDCS_Symptom_Score"),
                    ("hour_24_symptom_score", np.float64, "24_Hour_Symptom_Score"),
                    ("day_7_symptom_score", np.float64, "7_Day_Symptom_Score"),
                    ]
    feature_descr = [("age", np.float64, "Age"),
                     ("gender", gender_values, "Gender"),
                     ("pre_IL6", np.float64, "Pre_IL6"),
                     ("pre_TNF_Alpha", np.float64, "Pre_TNF_Alpha"),
                     ("pre_IgE", np.float64, "Pre_IgE"),
                     ("pre_symptom_score", np.float64, "Pre_Symptom_Score"),
                     ]

    ret = _fetch_remote_csv(ARCHIVE, "tDCS",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=18
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
