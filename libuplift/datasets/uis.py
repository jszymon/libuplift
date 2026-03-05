"""The UIS dataset from R quantreg package.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata

ARCHIVE = RemoteFileMetadata(
    filename=None, url=('local:uis_data'), checksum=None)


def fetch_uis(data_home=None, download_if_missing=True,
              random_state=None, shuffle=False,
              categ_as_strings=False, return_X_y=False,
              as_frame=False):
    """Load the UIS (Unemployment Insurance Study) drug treatment dataset.

    Use a local copy of the data.

    This dataset comes from a randomized clinical trial comparing short
    (3 months) vs long (6 months) drug treatment programs for substance
    abuse. The main outcome is time to drug relapse.

    Original source of the data is [1]_.  The version used here is
    from the R quantreg package [2]_.

    
    **Treatment Variables**
    
    - treatment: Treatment assignment (0 = Short, 1 = Long)

    **Target Variables**
    
    - target_time: Time to drug relapse (days)
    - target_censor: Censoring status (1 = Returned to drugs or lost to follow-up, 0 = Otherwise)
    - target_log_time: Log of time to drug relapse (Y variable)
    - target_FRAC: Compliance fraction (LEN.T/90 for short treatment,
      LEN.T/180 for long treatment).  This is included as another
      target variable since it is a post-randomization variable and
      sould not be used as predictor

    **Variables**

    - AGE: Age at Enrollment (Years)
    - BECK: Beck Depression Score (0.000 - 54.000)
    - HC: Heroin/Cocaine Use During 3 Months Prior to Admission
        * 1 = Heroin & Cocaine
        * 2 = Heroin Only
        * 3 = Cocaine Only
        * 4 = Neither Heroin nor Cocaine
    - IV: History of IV Drug Use
        * 1 = Never
        * 2 = Previous
        * 3 = Recent
    - NDT: Number of Prior Drug Treatments (0 - 40)
    - RACE: Subject's Race (0 = White, 1 = Non-White)
    - SITE: Treatment Site (0 = A, 1 = B)
    - LEN.T: Length of Stay in Treatment (Days)
    - ND1: Component of NDT
    - ND2: Component of NDT
    - LNDT: (Description not provided in original documentation)
    - IV3: Recent IV use (1 = Yes, 0 = No)

    **Changes to the original dataset**

    - remove the ID attribute

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
        object. Note that this returns a tuple with multiple target variables.

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

    (data, target_time, target_censor, target_log_time, treatment) : tuple if
        ``return_X_y`` is True

    References
    ----------

    .. [1] S.M. Hammer, et al., "A Controlled Trial of Two Nucleoside
       Analogues plus Indinavir in Persons with Human Immunodeficiency
       Virus Infection and CD4 Cell Counts of 200 per Cubic Millimeter
       or Less", New England Journal of Medicine, 337(11), 725--733,
       1997 (https://www.nejm.org/doi/10.1056/NEJMoa040595).

    .. [2]Koenker, R. (2022). quantreg: Quantile Regression. R package
       version 5.94.  https://CRAN.R-project.org/package=quantreg

    """

    # dictionaries for categorical variables
    hc_values = ["1", "2", "3", "4"]  # Heroin/Cocaine use
    iv_values = ["1", "2", "3"]      # IV drug use history

    # attribute descriptions
    treatment_descr = [("treatment", np.int32, "TREAT"),
                       ]
    
    target_descr = [("target_time", float, "TIME"),
                    ("target_censor", np.int32, "CENSOR"),
                    ("target_log_time", float, "Y"),
                    ("target_FRAC", float, "FRAC")]

    feature_descr = [("AGE", float),
                     ("BECK", float),
                     ("HC", hc_values),
                     ("IV", iv_values),
                     ("NDT", float),
                     ("RACE", np.int32),
                     ("SITE", np.int32),
                     ("LEN.T", float),
                     ("ND1", float),
                     ("ND2", float),
                     ("LNDT", float),
                     ("IV3", np.int32)]

    ret = _fetch_remote_csv(ARCHIVE, "uis",
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
