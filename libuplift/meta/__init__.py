"""Meta models: uplift models based on standard
classifiers/regressors."""

from .multi_model import MultimodelUpliftRegressor
from .multi_model import MultimodelUpliftClassifier
from .multi_model import TLearnerUpliftRegressor
from .multi_model import TLearnerUpliftClassifier
from .multi_model import MultimodelUpliftLinearRegressor
from .response import TreatmentUpliftClassifier
from .response import ResponseUpliftClassifier
from .response import ControlUpliftClassifier
from .target_transform import TargetTransformUpliftRegressor
from .target_transform import TargetTransformUpliftClassifier
from .s_learner import SLearnerUpliftRegressor
from .s_learner import SLearnerUpliftClassifier
from .nested import NestedMeanUpliftRegressor
from .nested import DDRUpliftClassifier
from .x_learner import XLearnerUpliftRegressor

__all__ = ["MultimodelUpliftRegressor",
           "MultimodelUpliftClassifier",
           "TLearnerUpliftRegressor",
           "TLearnerUpliftClassifier",
           "MultimodelUpliftLinearRegressor",
           "TreatmentUpliftClassifier",
           "ResponseUpliftClassifier",
           "ControlUpliftClassifier",
           "TargetTransformUpliftRegressor",
           "TargetTransformUpliftClassifier",
           "SLearnerUpliftRegressor",
           "SLearnerUpliftClassifier",
           "NestedMeanUpliftRegressor",
           "DDRUpliftClassifier",
           ]
