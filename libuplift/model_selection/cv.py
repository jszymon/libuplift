from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder

def uplift_check_cv(cv, y, trt, n_trt, *, classifier=False):
    """Return a correct cv and y_stratify.
    
    y_stratify may be used for stratification.  By default stratification
    is done based on treatment for regression and based on cross of
    treatment and target for classification.
    """
    
    # always stratify on treatment and, if available, also on class
    if classifier:
        le = LabelEncoder()
        y_stratify = le.fit_transform(y)
        y_stratify = y_stratify * (n_trt+1) + trt
    else:
        y_stratify = trt
    # classifier=True ensures stratification
    cv = check_cv(cv, y_stratify, classifier=True)
    return cv, y_stratify
