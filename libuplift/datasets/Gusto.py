"""The GUSTO-I clinical trial dataset.

This is a randomized clinical trial of thrombolytic strategies for
acute myocardial infractions.

This version is from the predtools R package.  See:
https://cran.r-project.org/web/packages/predtools/index.html for
details.

The study results were originally published in
https://www.nejm.org/doi/10.1056/NEJM199309023291001

The specific dataset used here together with a logistic model is
described in
https://www.ahajournals.org/doi/full/10.1161/01.cir.91.6.1659 and in
the book
https://www.clinicalpredictionmodels.org/extra-material/chapter-22

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="GUSTO.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/GUSTO/GUSTO.csv.gz'),
    checksum=('0f9ce4c74769ccc110bebe6a1f8beaee'
              '255c4b49811d8fea2e6eac0397da8bd9'))


def fetch_GUSTO(include_location_vars=True, include_ttr=False,
                data_home=None,
                download_if_missing=True,
                random_state=None, shuffle=False,
                categ_as_strings=False, return_X_y=False,
                as_frame=False):
    """Load the GUSTO-I clinical trial dataset.

    Download it if necessary.

    This is a randomized clinical trial dataset of thrombolytic
    strategies for acute myocardial infractions.

    WARNING: the ttr attribute breaks randomization, it is exluded by
    default, can be added using include_ttr=True e.g. to get nonrandom
    assignment which is easy to control.

    This version come from the predtools R package.  See:
    https://cran.r-project.org/web/packages/predtools/index.html for
    details.
    
    The study results were originally published in
    https://www.nejm.org/doi/10.1056/NEJM199309023291001
    
    The specific dataset used here together with a logistic model is
    described in
    https://www.ahajournals.org/doi/full/10.1161/01.cir.91.6.1659 and in
    the book
    https://www.clinicalpredictionmodels.org/extra-material/chapter-22

    **Variables**

    day30 (target)
        death within 30 days
    sho
        whether cardiac shock was present
    hig
        indicator of non-anterior MI location
    dia
        diabetes
    hyp
        high blood pressure indicator, seems to indicate sysbp >= 100 (5 exceptions)
    hrt
        tachycardia (indicator of pulse>80)
    ttr
        Time To Relief of chest pain > 1h
    sex
        patient's sex
    Killip
        Killip Class (I, II, III, IV)
    age
        patient's age [years]
    ste
        mumber of ECG leads with ST Elevation
    pulse
        Heart Rate [beats/min]
    sysbp
        Systolic Blood Pressure [mmHg]
    miloc
        MI Location (Inferior, Anterior, Other)
    height
        patient's height [cm]
    weight
        patient's weight [ckg]
    pmi
        previous MI
    htn
        history of hypertension
    smk
        smoking (never quit current)
    pan
        previous angina pectoris
    fam
        family history of MI
    prevcvd
        previous cardiovascular disease
    prevcabg
        previous coronary artery bypass graft surgery
    regl
        region (probably country)
    grpl
        location code 2, refinement of regl
    grps
        location code 3, refinement of regl
    tx
        treatment (SK, SK+tPA, tPA)

    **Changes to the original dataset**

    - reverse the hyp indicator variable such that 1 corresponds to sysbp >= 100
    - removed tpa an indicator of tPA treatment (can be inferred from tx)
    - removed ant variable which is an indicator anterior MI (included in miloc)
    - change pmi to {0,1} binary indicator
    - subtract 1 from htn to make it {0,1} binary indicator
    - subtract 1 from pan to make it {0,1} binary indicator
    - subtract 1 from fam to make it {0,1} binary indicator

    **Functional dependencies**

    - sho can be inferred from Killip
    - hrt can be inferred from pulse
    - hyp can be inferred from sysbp >= 100 (except 5 cases)
    - hig: there is a functional dependency: hig=0 -> miloc=Anterior
    - grpl is a refinement regl
    - grps is a refinement regl

    Parameters
    ----------
    include_location_vars : boolean, default=True
        Should variables describing hospital locations be
        included. These are categorical variables with large number of
        levels.  The removed variables are regl, grpl, grps
    
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

    (data, target) : tuple if ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ["SK", "SK+tPA", "tPA"]
    sex_values = ["male", "female"]
    Killip_values = ["I", "II", "III", "IV"]
    miloc_values = ["Inferior", "Anterior", "Other"]
    smk_values = ["never", "quit", "current"]
    regl_values = [str(i+1) for i in range(16)]
    grpl_values = [str(i+1) for i in range(48)]
    grps_values = [str(i+1) for i in range(121)]

    # attribute descriptions
    treatment_descr = [("treatment", treatment_values, "tx")]
    target_descr = [("target", np.int32, "day30"),
                    ]
    feature_descr = [('sho', np.int32),
                     ('hig', np.int32),
                     ('dia', np.int32),
                     ('hyp', np.int32),
                     ('hrt', np.int32),
                     ('ttr', np.int32),
                     ('sex', sex_values),
                     ('Killip', Killip_values),
                     ('age', float),
                     ('ste', np.int32),
                     ('pulse', np.int32),
                     ('sysbp', np.int32),
                     ('miloc', miloc_values),
                     ('height', float),
                     ('weight', float),
                     ('pmi', np.int32),
                     ('htn', np.int32),
                     ('smk', smk_values),
                     ('pan', np.int32),
                     ('fam', np.int32),
                     ('prevcvd', np.int32),
                     ('prevcabg', np.int32),
                     ('regl', regl_values),
                     ('grpl', grpl_values),
                     ('grps', grps_values),
                     ]

    arch = ARCHIVE
    dataset_name = "GUSTO"
    remove_vars=[]
    if not include_ttr:
        remove_vars += ['ttr']
    if not include_location_vars:
        remove_vars += ['regl', 'grpl', 'grps']
    if len(remove_vars) == 0:
        remove_vars = None
    ret = _fetch_remote_csv(arch, dataset_name,
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=27, all_num=False,
                            remove_vars=remove_vars
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
