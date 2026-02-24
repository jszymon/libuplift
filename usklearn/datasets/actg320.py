"""The actg320 trial data from Hosmer, Lemeshow and May.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="actg320.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/actg320/actg320.csv.gz'),
    checksum=('bd39c16f8c15b2dada38d6af702bbcf6'
              '4ccb33f0a5a8d8a95ee001a710f444c7'))

def fetch_actg320(data_home=None,
                  download_if_missing=True,
                  random_state=None, shuffle=False,
                  categ_as_strings=False, return_X_y=False,
                  as_frame=False):
    """Load the actg320 AIDS treatment clinical trial dataset.

    Download it if necessary.

    This is a randomized clinical trial dataset of various AIDS
    treatments from [2]_.

    The description of the original study can be found in [1]_.

    The main treatment variable indicates whether treatment includes
    IDV (Indinavir).  The treatment_grp variable contains one of four
    specific treatments given:
    1 = ZDV + 3TC
    2 = ZDV + 3TC + IDV
    3 = d4T + 3TC 
    4 = d4T + 3TC + IDV
    (treatments 3 and 4 were given in only 3 cases)

    Treatment assignment was stratified on strat2 variable (CD4 count).

    Target variables:
    time/censor: time/censoring to occurrence of AIDS or death
    time_d/censor_d: time/censoring to occurrence of death

    **Variables**

    strat2
        CD4 stratum at screening 0: CD4 <= 50, 1: > 50
    sex
        1 = Male, 2 = Female 
    raceth
        Race/Ethnicity:
        1 = White Non-Hispanic
        2 = Black Non-Hispanic
        3 = Hispanic (regardless of race)
        4 = Asian, Pacific Islander 
        5 = American Indian, Alaskan Native
        6 = Other/unknown
    ivdrug
        IV drug use history:
        1 = Never
        2 = Currently
        3 = Previously
    hemophil
        Hemophiliac
    karnof
        Karnofsky Performance Scale
    cd4
        Baseline CD4 count [Cells/milliliter]
    priorzdv
        Months of prior ZDV use [months]
    age
        Age at Enrollment [years]
    
    Parameters
    ----------
    include_location_vars : boolean, default=True
        Should variables describing hospital locations be
        included. These are categorical variables with large number of
        levels.  The removed variables are regl, grpl, grps
    
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
        Each value is 1 if a purchase was made 0 otherwise.

    dataset.DESCR : string
        Description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    References
    ----------

    .. [1] S.M. Hammer, et al., "A Controlled Trial of Two Nucleoside
       Analogues plus Indinavir in Persons with Human Immunodeficiency
       Virus Infection and CD4 Cell Counts of 200 per Cubic Millimeter
       or Less", New England Journal of Medicine, 337(11), 725--733,
       1997 (https://www.nejm.org/doi/10.1056/NEJM199709113371101)

    .. [2] Hosmer, D.W. and Lemeshow, S. and May, S.,  Applied
       Survival Analysis: Regression Modeling of Time to Event Data:
       Second Edition, John Wiley and Sons Inc., New York, NY, 2008.

    """
    # dictionaries
    karnof_values = ["70", "80", "90", "100"]

    # attribute descriptions
    treatment_descr = [("treatment", np.int32, "tx"),
                       ("treatment_grp", np.int32, "txgrp"),
                       ]
    target_descr = [("target_time", float, "time"),
                    ("target_censor", np.int32, "censor"),
                    ("target_time_d", float, "time_d"),
                    ("target_censor_d", np.int32, "censor_d"),
                    ]
    feature_descr = [("strat2", np.int32),
                     ("sex", ["1","2"]),
                     ("raceth", ["1","2","3","4","5"]),
                     ("ivdrug", ["1","2","3"]),
                     ("hemophil", np.int32),
                     ("karnof", karnof_values),
                     ("cd4", float),
                     ("priorzdv", float),
                     ("age", float),
                     ]

    ret = _fetch_remote_csv(ARCHIVE, "actg320",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=15
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
