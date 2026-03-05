"""Multiple simultaneuosly indexed arrays.

Needed to work around lack of sample properties in scikit-learn."""

import numpy as np

from sklearn.utils import check_consistent_length

class MultiArray:
    __array_ufunc__ = None # don't allow numpy operations by default

    def __init__(self, main_array, array_dict=None, scalar_dict=None):
        """behaves like main_array w.r.t. indexing, but arrays in

        array_dict are indexed simulteneously, scalar_dict is passed to
        indexing results."""
        self.main_array = main_array
        if array_dict is None:
            array_dict = dict()
        check_consistent_length([main_array] + list(array_dict.values()))
        self.array_dict = array_dict
        if scalar_dict is None:
            scalar_dict = dict()
        self.scalar_dict = scalar_dict
        self.shape = self.main_array.shape
        self.ndim = self.main_array.ndim
        self.dtype = self.main_array.dtype
    def __getitem__(self, idx):
        new_dict = {k:self.array_dict[k][idx] for k in self.array_dict}
        return MultiArray(self.main_array[idx], new_dict, self.scalar_dict)
