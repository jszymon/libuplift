"""
The :mod:`usklearn.utils` module includes various utilities.
"""

from .validation import check_trt
from .multi_array import MultiArray
from .array_utils import safe_hstack
from .metrics import area_under_curve

__all__ = ["check_trt", "MultiArray", "safe_hstack", "area_under_curve"]
