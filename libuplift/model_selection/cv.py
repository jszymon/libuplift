from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder

def uplift_check_cv(cv, y, trt, n_trt, *, classifier=False, y_stratify=None):
    """Return a correct crossvalidator cv and stratification target
    y_stratify.
    
    By default the returned stratification target is the treatment for
    regression and cross of treatment and target for classification.
    If y_stratify is provided it is used instead and returned
    unchanged.

    """
    
    if y_stratify is None:
        # always stratify on treatment and, if needed, also on class
        if classifier:
            le = LabelEncoder()
            y_stratify = le.fit_transform(y)
            y_stratify = y_stratify * (n_trt+1) + trt
        else:
            y_stratify = trt.copy()
    # classifier=True ensures stratification
    cv = check_cv(cv, y_stratify, classifier=True)
    return cv, y_stratify
