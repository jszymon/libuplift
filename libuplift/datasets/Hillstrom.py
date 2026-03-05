"""Hillstrom challenge dataset.

A dataset from Kevin Hillstrom's MineThatData blog. See

    https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

for details.
"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="Hillstrom.csv",
    url=('http://www.minethatdata.com/'
         'Kevin_Hillstrom_MineThatData_E-MailAnalytics'
         '_DataMiningChallenge_2008.03.20.csv'),
    checksum=('0e5893329d8b93cefecc571777672028'
              '290ab69865718020c78c7284f291aece'))


def fetch_Hillstrom(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False,
                    as_frame=False):
    """Load the Hillstrom dataset (uplift classification and regression).

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

    dataset.target_visit : numpy array
        Each value is 1 if website visit occurred 0 otherwise.

    dataset.target_conversion : numpy array
        Each value is 1 if purchase occurred 0 otherwise.

    dataset.target_spend : numpy array
        Each value corresponds to the amount of money spent.

    dataset.DESCR : string
        Description of the Hillstrom dataset.

    (data, target_visit, target_conversion, target_spend) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['No E-Mail', 'Mens E-Mail', 'Womens E-Mail']
    history_segment_values = ['1) $0 - $100', '2) $100 - $200',
                              '3) $200 - $350', '4) $350 - $500',
                              '5) $500 - $750', '6) $750 - $1,000',
                              '7) $1,000 +']
    zip_code_values = ['Rural', 'Surburban', 'Urban']
    channel_values = ['Phone', 'Web', 'Multichannel']
    categ_values = {"history_segment": history_segment_values,
                    "zip_code": zip_code_values,
                    "channel": channel_values,}

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "segment")]
    target_descr = [("target_visit", np.int32, "visit"),
                    ("target_conversion", np.int32, "conversion"),
                    ("target_spend", float, "spend")]
    feature_descr = [("recency", np.int32),
                     ("history_segment", history_segment_values),
                     ("history", float),
                     ("mens", np.int32),
                     ("womens", np.int32),
                     ("zip_code", zip_code_values),
                     ("newbie", np.int32),
                     ("channel", channel_values)]

    ret = _fetch_remote_csv(ARCHIVE, "Hillstrom",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=12
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
