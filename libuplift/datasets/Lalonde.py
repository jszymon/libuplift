"""The Lalonde (jobs) datasets.

There are two versions of the dataset, including 1974 earnings
(version B) and excluding 1974 earnings (vestion A).
"""


from os.path import dirname, exists, join
from os import remove, makedirs
import csv

import numpy as np

from sklearn.datasets import get_data_home
from .base import _fetch_remote_csv
from .base import _prepare_final_data
from .base import RemoteFileMetadata
from sklearn.utils import Bunch
import joblib
from sklearn.utils import check_random_state

ARCHIVE_A_T = RemoteFileMetadata(
    filename="Lalonde_A_T.txt",
    url='http://www.nber.org/~rdehejia/data/nsw_treated.txt',
    checksum=('ab65cd58de17a78b692e66e4d7142192'
              '59ac180428f24c42ddbb928cfb1820fe'))
ARCHIVE_A_C = RemoteFileMetadata(
    filename="Lalonde_A_C.txt",
    url='http://www.nber.org/~rdehejia/data/nsw_control.txt',
    checksum=('8fd745ed2c3426bb77e34b395fb84d45'
              '6d346ba545af58a65b4963160b0699fd'))

ARCHIVE_B_T = RemoteFileMetadata(
    filename="Lalonde_B_T.txt",
    url='http://www.nber.org/~rdehejia/data/nswre74_treated.txt',
    checksum=('e7b742fe0ff07a0f45e129b4ff108bb9'
              '611cd83d53604732c48a8a0a3e20eda3'))
ARCHIVE_B_C = RemoteFileMetadata(
    filename="Lalonde_B_C.txt",
    url='http://www.nber.org/~rdehejia/data/nswre74_control.txt',
    checksum=('a1364cea459d953dc691a667d99194b4'
              'ad335d6d550354fe23a5d2dc58d729b5'))


def fetch_Lalonde(version="A", data_home=None,
                  categ_as_strings=False,
                  download_if_missing=True, random_state=None,
                  shuffle=False, return_X_y=False,
                  as_frame=False):
    """Load the Lalonde datasets (uplift regression).

    Download it if necessary.

    There are two versions of the dataset, including 1974 earnings
    (version B) and excluding 1974 earnings (vestion A).
    Source: http://users.nber.org/~rdehejia/data/nswdata2.html

    Parameters
    ----------
    version : string, optional
        Specify which dataset to return.  'A' for larger files
        without 1974 earnings, 'B' for smaller files with 1974
        earnings.

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

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest covertypes with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest covertype dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """

      
    #data_home = get_data_home(data_home=data_home)
    #Lalonde_dir = join(data_home, "uplift_sklearn", "Lalonde")
    #samples_path = join(Lalonde_dir, "samples" + version_suffix)
    #targets_path = join(Lalonde_dir, "targets" + version_suffix)
    #treatment_path = join(Lalonde_dir, "treatment" + version_suffix)
    #available = exists(samples_path)

    # dictionaries
    header_A = ['treatment', 'age', 'education', 'Black', 'Hispanic',
                'married', 'nodegree', 'RE75', 'RE78']
    header_B = ['treatment', 'age', 'education', 'Black', 'Hispanic',
                'married', 'nodegree', 'RE74', 'RE75', 'RE78']

    target_names = ["RE78"]

    def _float_to_int(x):
        return np.array(x, float), np.int32
    
    # attribute descriptions
    treatment_descr = [("treatment", _float_to_int)]
    target_descr = [("target_RE78", float, "RE78")]
    feature_descr_all = [("age", float),
                         ("education", float),
                         ("Black", float),
                         ("Hispanic", float),
                         ("married", float),
                         ("nodegree", float)]
    feature_descr_A = feature_descr_all + [("RE75", float)]
    feature_descr_B = feature_descr_all + [("RE74", float), ("RE75", float)]
    csv_reader_args = {"delimiter":' ', "skipinitialspace":True}
    
    # choose version
    if version == "A":
        version_suffix = "_A"
        arch_t = ARCHIVE_A_T
        arch_c = ARCHIVE_A_C
        n_fields = 9
        header = header_A
        feature_descr = feature_descr_A
    elif version == "B":
        version_suffix = "_B"
        arch_t = ARCHIVE_B_T
        arch_c = ARCHIVE_B_C
        n_fields = 10
        header = header_B
        feature_descr = feature_descr_B
    else:
        raise ValueError("Lalonde dataset version must be A or B")



    D_T = _fetch_remote_csv(arch_t, "Lalonde_T"+version_suffix,
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=False, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=False,
                            total_attrs=n_fields, header=header,
                            csv_reader_args=csv_reader_args
                            )
    assert np.all(D_T.treatment == 1)
    D_C = _fetch_remote_csv(arch_c, "Lalonde_C"+version_suffix,
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=False, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=False,
                            total_attrs=n_fields, header=header,
                            csv_reader_args=csv_reader_args
                            )
    assert np.all(D_C.treatment == 0)

    # combine treatment and control datasets
    D = D_C
    if as_frame:
        import pandas
        D.data = pandas.concat([D_C.data, D_T.data], ignore_index=True)
    else:
        D.data = np.concatenate([D_C.data, D_T.data])
    D.treatment = np.concatenate([D_C.treatment, D_T.treatment])
    D.target_RE78 = np.concatenate([D_C.target_RE78, D_T.target_RE78])

    # prepare final return value
    ret = _prepare_final_data(D, shuffle, return_X_y)
    if not return_X_y:
        ret.descr = __doc__
    return ret
