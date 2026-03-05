"""The marketing campaign dataset from the CRAN Information package by
Kim Larsen.

See: https://cran.r-project.org/web/packages/Information/index.html
for details.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE_T = RemoteFileMetadata(
    filename="Information_train.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Information/Information_train.csv.gz'),
    checksum=('7632536786357871de3f2438dbbe6c70'
              '17829b2d244b2878dba9fb11af0a449b'))
ARCHIVE_V = RemoteFileMetadata(
    filename="Information_valid.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Information/Information_valid.csv.gz'),
    checksum=('96b1773bc9fea564de03d412b936b32a'
              'b5d3b1a9f7338dbb59aa6f918806fe66'))


def fetch_Information(version="train", data_home=None,
                      download_if_missing=True,
                      random_state=None, shuffle=False,
                      categ_as_strings=False, return_X_y=False,
                      as_frame=False):
    """Load the marketing campaign dataset from the CRAN Information
    package by Kim Larsen.

    See: https://cran.r-project.org/web/packages/Information/index.html

    Two datasets are available: "train" and "validation".  Use version
    argument to select.
    
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

    dataset.target : numpy array
        Each value is 1 if a purchase was made 0 otherwise.

    dataset.DESCR : string
        Description of the dataset.

    (data, target_conversion, target_visit, target_exposure) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['control', 'treated']
    categ_values = dict()

    # attribute descriptions
    treatment_descr = [("treatment", np.int32, "TREATMENT")]
    target_descr = [("target", np.int32, "PURCHASE"),
                    ]
    feature_descr = [('AGE', float),
                     ('AGRGT_BAL_ALL_XCLD_MRTG', float),
                     ('AUTO_2_OPEN_DATE_YRS', float),
                     ('AUTO_HI_CRDT_2_ACTUAL', float),
                     ('AVG_BAL_ALL_FNC_REV_ACTS', float),
                     ('AVG_BAL_ALL_PRM_BC_ACTS', float),
                     ('D_DEPTCARD', float),
                     ('D_NA_AVG_BAL_ALL_FNC_REV_ACTS', float),
                     ('D_NA_M_SNCOLDST_BNKINSTL_ACTOPN', float),
                     ('D_NA_M_SNC_MST_RCNT_ACT_OPN', float),
                     ('D_NA_M_SNC_MST_RCNT_MRTG_DEAL', float),
                     ('D_NA_M_SNC_OLDST_MRTG_ACT_OPN', float),
                     ('D_NA_RATIO_PRSNL_FNC_BAL2HICRDT', float),
                     ('D_REGION_A', float),
                     ('D_REGION_B', float),
                     ('D_REGION_C', float),
                     ('FNC_CARD_OPEN_DATE_YRS', float),
                     ('HI_RETAIL_CRDT_LMT', float),
                     ('MAX_MRTG_CLOSE_DATE', float),
                     ('MRTG_1_MONTHLY_PAYMENT', float),
                     ('MRTG_2_CURRENT_BAL', float),
                     ('M_SNCOLDST_BNKINSTL_ACTOPN', float),
                     ('M_SNCOLDST_OIL_NTN_TRD_OPN', float),
                     ('M_SNC_MSTRCNT_MRTG_ACT_UPD', float),
                     ('M_SNC_MSTREC_INSTL_TRD_OPN', float),
                     ('M_SNC_MST_RCNT_60_DAY_RTNG', float),
                     ('M_SNC_MST_RCNT_ACT_OPN', float),
                     ('M_SNC_MST_RCNT_MRTG_DEAL', float),
                     ('M_SNC_OLDST_MRTG_ACT_OPN', float),
                     ('M_SNC_OLDST_RETAIL_ACT_OPN', float),
                     ('N30D_ORWRS_RTNG_MRTG_ACTS', float),
                     ('N_120D_RATINGS', float),
                     ('N_30D_AND_60D_RATINGS', float),
                     ('N_30D_RATINGS', float),
                     ('N_ACTS_90D_PLS_LTE_IN_6M', float),
                     ('N_ACTS_WITH_MXD_3_IN_24M', float),
                     ('N_ACTS_WITH_MXD_4_IN_24M', float),
                     ('N_BANK_INSTLACTS', float),
                     ('N_BC_ACTS_OPN_IN_12M', float),
                     ('N_BC_ACTS_OPN_IN_24M', float),
                     ('N_DEROG_PUB_RECS', float),
                     ('N_DISPUTED_ACTS', float),
                     ('N_FNC_ACTS_OPN_IN_12M', float),
                     ('N_FNC_ACTS_VRFY_IN_12M', float),
                     ('N_FNC_INSTLACTS', float),
                     ('N_INQUIRIES', float),
                     ('N_OF_MRTG_ACTS_DLINQ_24M', float),
                     ('N_OF_SATISFY_FNC_REV_ACTS', float),
                     ('N_OPEN_REV_ACTS', float),
                     ('N_PUB_REC_ACT_LINE_DEROGS', float),
                     ('N_RETAIL_ACTS_OPN_IN_24M', float),
                     ('N_SATISFY_INSTL_ACTS', float),
                     ('N_SATISFY_OIL_NATIONL_ACTS', float),
                     ('N_SATISFY_PRSNL_FNC_ACTS', float),
                     ('PRCNT_OF_ACTS_NEVER_DLQNT', float),
                     ('PREM_BANKCARD_CRED_LMT', float),
                     ('RATIO_BAL_TO_HI_CRDT', float),
                     ('RATIO_PRSNL_FNC_BAL2HICRDT', float),
                     ('RATIO_RETAIL_BAL2HI_CRDT', float),
                     ('STUDENT_HI_CRED_RANGE', float),
                     ('STUDENT_OPEN_DATE_YRS', float),
                     ('TOT_BAL_ALL_DPT_STORE_ACTS', float),
                     ('TOT_HI_CRDT_CRDT_LMT', float),
                     ('TOT_INSTL_HI_CRDT_CRDT_LMT', float),
                     ('TOT_NOW_LTE', float),
                     ('TOT_OTHRFIN_HICRDT_CRDTLMT', float),
                     ('UPSCALE_OPEN_DATE_YRS', float),
                     ]


    if version == "train":
        arch = ARCHIVE_T
    elif version == "valid":
        arch = ARCHIVE_V
    else:
        raise ValueError("Wrong Information dataset version."
                         "  Got '{version}', expected 'train' or 'valid'.")
    ret = _fetch_remote_csv(arch, "Information",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=69, all_num=True
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
