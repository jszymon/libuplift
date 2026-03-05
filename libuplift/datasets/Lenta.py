"""Lenta challenge dataset.

The version used is from the scikit-uplift package with non-ascii
characters removed from the Gender variable.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="lenta_dataset.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Lenta/lenta_dataset.csv.gz'),
    checksum=('9002bc5c52ab64c2d68f517cb3390756'
              '87a380d7b407eef1926bd9eda5f31e5d'))

def _float_w_nan(x):
    """Convert strings to floats with empty strings converted to
    nan's."""
    y = [v if v != "" else "nan" for v in x]
    return np.array(y, float), float

def fetch_Lenta(data_home=None, download_if_missing=True,
                random_state=None, shuffle=False,
                categ_as_strings=False, return_X_y=False,
                as_frame=False):
    """Load the Lenta dataset (uplift classification).

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

    dataset.target_response_att : numpy array
        Each value is 1 if store visit occurred 0 otherwise.

    dataset.DESCR : string
        Description of the dataset.

    (data, target_response_att) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['control', 'test']
    gender_values = ["F", "M", "Unspecified", "NA"]
    
    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "group")]
    target_descr = [("target_response_att", np.int32, "response_att")]

    feature_descr = [('age', _float_w_nan),
                     ('cheque_count_12m_g20', _float_w_nan),
                     ('cheque_count_12m_g21', _float_w_nan),
                     ('cheque_count_12m_g25', _float_w_nan),
                     ('cheque_count_12m_g32', _float_w_nan),
                     ('cheque_count_12m_g33', _float_w_nan),
                     ('cheque_count_12m_g38', _float_w_nan),
                     ('cheque_count_12m_g39', _float_w_nan),
                     ('cheque_count_12m_g41', _float_w_nan),
                     ('cheque_count_12m_g42', _float_w_nan),
                     ('cheque_count_12m_g45', _float_w_nan),
                     ('cheque_count_12m_g46', _float_w_nan),
                     ('cheque_count_12m_g48', _float_w_nan),
                     ('cheque_count_12m_g52', _float_w_nan),
                     ('cheque_count_12m_g56', _float_w_nan),
                     ('cheque_count_12m_g57', _float_w_nan),
                     ('cheque_count_12m_g58', _float_w_nan),
                     ('cheque_count_12m_g79', _float_w_nan),
                     ('cheque_count_3m_g20',  _float_w_nan),
                     ('cheque_count_3m_g21',  _float_w_nan),
                     ('cheque_count_3m_g25',  _float_w_nan),
                     ('cheque_count_3m_g42',  _float_w_nan),
                     ('cheque_count_3m_g45',  _float_w_nan),
                     ('cheque_count_3m_g52',  _float_w_nan),
                     ('cheque_count_3m_g56',  _float_w_nan),
                     ('cheque_count_3m_g57',  _float_w_nan),
                     ('cheque_count_3m_g79',  _float_w_nan),
                     ('cheque_count_6m_g20',  _float_w_nan),
                     ('cheque_count_6m_g21',  _float_w_nan),
                     ('cheque_count_6m_g25',  _float_w_nan),
                     ('cheque_count_6m_g32',  _float_w_nan),
                     ('cheque_count_6m_g33',  _float_w_nan),
                     ('cheque_count_6m_g38',  _float_w_nan),
                     ('cheque_count_6m_g39',  _float_w_nan),
                     ('cheque_count_6m_g40',  _float_w_nan),
                     ('cheque_count_6m_g41',  _float_w_nan),
                     ('cheque_count_6m_g42',  _float_w_nan),
                     ('cheque_count_6m_g45',  _float_w_nan),
                     ('cheque_count_6m_g46',  _float_w_nan),
                     ('cheque_count_6m_g48',  _float_w_nan),
                     ('cheque_count_6m_g52',  _float_w_nan),
                     ('cheque_count_6m_g56',  _float_w_nan),
                     ('cheque_count_6m_g57',  _float_w_nan),
                     ('cheque_count_6m_g58',  _float_w_nan),
                     ('cheque_count_6m_g79',  _float_w_nan),
                     ('children', _float_w_nan),
                     ('crazy_purchases_cheque_count_12m', _float_w_nan),
                     ('crazy_purchases_cheque_count_1m',  _float_w_nan),
                     ('crazy_purchases_cheque_count_3m',  _float_w_nan),
                     ('crazy_purchases_cheque_count_6m',  _float_w_nan),
                     ('crazy_purchases_goods_count_12m',  _float_w_nan),
                     ('crazy_purchases_goods_count_6m',   _float_w_nan),
                     ('disc_sum_6m_g34', _float_w_nan),
                     ('food_share_15d', _float_w_nan),
                     ('food_share_1m', _float_w_nan),
                     ('gender', gender_values),
                     ('k_var_cheque_15d', _float_w_nan),
                     ('k_var_cheque_3m', _float_w_nan),
                     ('k_var_cheque_category_width_15d', _float_w_nan),
                     ('k_var_cheque_group_width_15d', _float_w_nan),
                     ('k_var_count_per_cheque_15d_g24', _float_w_nan),
                     ('k_var_count_per_cheque_15d_g34', _float_w_nan),
                     ('k_var_count_per_cheque_1m_g24', _float_w_nan),
                     ('k_var_count_per_cheque_1m_g27', _float_w_nan),
                     ('k_var_count_per_cheque_1m_g34', _float_w_nan),
                     ('k_var_count_per_cheque_1m_g44', _float_w_nan),
                     ('k_var_count_per_cheque_1m_g49', _float_w_nan),
                     ('k_var_count_per_cheque_3m_g24', _float_w_nan),
                     ('k_var_count_per_cheque_3m_g27', _float_w_nan),
                     ('k_var_count_per_cheque_3m_g32', _float_w_nan),
                     ('k_var_count_per_cheque_3m_g34', _float_w_nan),
                     ('k_var_count_per_cheque_3m_g41', _float_w_nan),
                     ('k_var_count_per_cheque_3m_g44', _float_w_nan),
                     ('k_var_count_per_cheque_6m_g24', _float_w_nan),
                     ('k_var_count_per_cheque_6m_g27', _float_w_nan),
                     ('k_var_count_per_cheque_6m_g32', _float_w_nan),
                     ('k_var_count_per_cheque_6m_g44', _float_w_nan),
                     ('k_var_days_between_visits_15d', _float_w_nan),
                     ('k_var_days_between_visits_1m', _float_w_nan),
                     ('k_var_days_between_visits_3m', _float_w_nan),
                     ('k_var_disc_per_cheque_15d', _float_w_nan),
                     ('k_var_disc_share_12m_g32', _float_w_nan),
                     ('k_var_disc_share_15d_g24', _float_w_nan),
                     ('k_var_disc_share_15d_g34', _float_w_nan),
                     ('k_var_disc_share_15d_g49', _float_w_nan),
                     ('k_var_disc_share_1m_g24', _float_w_nan),
                     ('k_var_disc_share_1m_g27', _float_w_nan),
                     ('k_var_disc_share_1m_g34', _float_w_nan),
                     ('k_var_disc_share_1m_g40', _float_w_nan),
                     ('k_var_disc_share_1m_g44', _float_w_nan),
                     ('k_var_disc_share_1m_g49', _float_w_nan),
                     ('k_var_disc_share_1m_g54', _float_w_nan),
                     ('k_var_disc_share_3m_g24', _float_w_nan),
                     ('k_var_disc_share_3m_g26', _float_w_nan),
                     ('k_var_disc_share_3m_g27', _float_w_nan),
                     ('k_var_disc_share_3m_g32', _float_w_nan),
                     ('k_var_disc_share_3m_g33', _float_w_nan),
                     ('k_var_disc_share_3m_g34', _float_w_nan),
                     ('k_var_disc_share_3m_g38', _float_w_nan),
                     ('k_var_disc_share_3m_g40', _float_w_nan),
                     ('k_var_disc_share_3m_g41', _float_w_nan),
                     ('k_var_disc_share_3m_g44', _float_w_nan),
                     ('k_var_disc_share_3m_g46', _float_w_nan),
                     ('k_var_disc_share_3m_g48', _float_w_nan),
                     ('k_var_disc_share_3m_g49', _float_w_nan),
                     ('k_var_disc_share_3m_g54', _float_w_nan),
                     ('k_var_disc_share_6m_g24', _float_w_nan),
                     ('k_var_disc_share_6m_g27', _float_w_nan),
                     ('k_var_disc_share_6m_g32', _float_w_nan),
                     ('k_var_disc_share_6m_g34', _float_w_nan),
                     ('k_var_disc_share_6m_g44', _float_w_nan),
                     ('k_var_disc_share_6m_g46', _float_w_nan),
                     ('k_var_disc_share_6m_g49', _float_w_nan),
                     ('k_var_disc_share_6m_g54', _float_w_nan),
                     ('k_var_discount_depth_15d', _float_w_nan),
                     ('k_var_discount_depth_1m', _float_w_nan),
                     ('k_var_sku_per_cheque_15d', _float_w_nan),
                     ('k_var_sku_price_12m_g32', _float_w_nan),
                     ('k_var_sku_price_15d_g34', _float_w_nan),
                     ('k_var_sku_price_15d_g49', _float_w_nan),
                     ('k_var_sku_price_1m_g24', _float_w_nan),
                     ('k_var_sku_price_1m_g26', _float_w_nan),
                     ('k_var_sku_price_1m_g27', _float_w_nan),
                     ('k_var_sku_price_1m_g34', _float_w_nan),
                     ('k_var_sku_price_1m_g40', _float_w_nan),
                     ('k_var_sku_price_1m_g44', _float_w_nan),
                     ('k_var_sku_price_1m_g49', _float_w_nan),
                     ('k_var_sku_price_1m_g54', _float_w_nan),
                     ('k_var_sku_price_3m_g24', _float_w_nan),
                     ('k_var_sku_price_3m_g26', _float_w_nan),
                     ('k_var_sku_price_3m_g27', _float_w_nan),
                     ('k_var_sku_price_3m_g32', _float_w_nan),
                     ('k_var_sku_price_3m_g33', _float_w_nan),
                     ('k_var_sku_price_3m_g34', _float_w_nan),
                     ('k_var_sku_price_3m_g40', _float_w_nan),
                     ('k_var_sku_price_3m_g41', _float_w_nan),
                     ('k_var_sku_price_3m_g44', _float_w_nan),
                     ('k_var_sku_price_3m_g46', _float_w_nan),
                     ('k_var_sku_price_3m_g48', _float_w_nan),
                     ('k_var_sku_price_3m_g49', _float_w_nan),
                     ('k_var_sku_price_3m_g54', _float_w_nan),
                     ('k_var_sku_price_6m_g24', _float_w_nan),
                     ('k_var_sku_price_6m_g26', _float_w_nan),
                     ('k_var_sku_price_6m_g27', _float_w_nan),
                     ('k_var_sku_price_6m_g32', _float_w_nan),
                     ('k_var_sku_price_6m_g41', _float_w_nan),
                     ('k_var_sku_price_6m_g42', _float_w_nan),
                     ('k_var_sku_price_6m_g44', _float_w_nan),
                     ('k_var_sku_price_6m_g48', _float_w_nan),
                     ('k_var_sku_price_6m_g49', _float_w_nan),
                     ('main_format', _float_w_nan),
                     ('mean_discount_depth_15d', _float_w_nan),
                     ('months_from_register', _float_w_nan),
                     ('perdelta_days_between_visits_15_30d', _float_w_nan),
                     ('promo_share_15d', _float_w_nan),
                     ('response_sms', _float_w_nan),
                     ('response_viber', _float_w_nan),
                     ('sale_count_12m_g32', _float_w_nan),
                     ('sale_count_12m_g33', _float_w_nan),
                     ('sale_count_12m_g49', _float_w_nan),
                     ('sale_count_12m_g54', _float_w_nan),
                     ('sale_count_12m_g57', _float_w_nan),
                     ('sale_count_3m_g24', _float_w_nan),
                     ('sale_count_3m_g33', _float_w_nan),
                     ('sale_count_3m_g57', _float_w_nan),
                     ('sale_count_6m_g24', _float_w_nan),
                     ('sale_count_6m_g25', _float_w_nan),
                     ('sale_count_6m_g32', _float_w_nan),
                     ('sale_count_6m_g33', _float_w_nan),
                     ('sale_count_6m_g44', _float_w_nan),
                     ('sale_count_6m_g54', _float_w_nan),
                     ('sale_count_6m_g57', _float_w_nan),
                     ('sale_sum_12m_g24', _float_w_nan),
                     ('sale_sum_12m_g25', _float_w_nan),
                     ('sale_sum_12m_g26', _float_w_nan),
                     ('sale_sum_12m_g27', _float_w_nan),
                     ('sale_sum_12m_g32', _float_w_nan),
                     ('sale_sum_12m_g44', _float_w_nan),
                     ('sale_sum_12m_g54', _float_w_nan),
                     ('sale_sum_3m_g24', _float_w_nan),
                     ('sale_sum_3m_g26', _float_w_nan),
                     ('sale_sum_3m_g32', _float_w_nan),
                     ('sale_sum_3m_g33', _float_w_nan),
                     ('sale_sum_6m_g24', _float_w_nan),
                     ('sale_sum_6m_g25', _float_w_nan),
                     ('sale_sum_6m_g26', _float_w_nan),
                     ('sale_sum_6m_g32', _float_w_nan),
                     ('sale_sum_6m_g33', _float_w_nan),
                     ('sale_sum_6m_g44', _float_w_nan),
                     ('sale_sum_6m_g54', _float_w_nan),
                     ('stdev_days_between_visits_15d', _float_w_nan),
                     ('stdev_discount_depth_15d', _float_w_nan),
                     ('stdev_discount_depth_1m', _float_w_nan),
                     ]


    ret = _fetch_remote_csv(ARCHIVE, "Lenta",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=195
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
