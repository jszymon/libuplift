"""The International Stroke Trial dataset.

This is a randomized clinical trial of heparin and aspirin treatment
for stroke patients.

This dataset is derived from the corrected dataset available here:
https://datashare.ed.ac.uk/handle/10283/128
The webpage contains detailed descriptions.

This version only includes pre-randomization variables, two targets,
and several additional targets related to side effects.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="IST.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/IST/IST.csv.gz'),
    checksum=('24401e85937748cb0488994c1d8aaf6e'
              'be34a5d12d5446840b4f038f8a8e4de7'))


def fetch_IST(include_pilot=True,
              include_location_vars=True,
              include_prediction_model_vars=True,
              data_home=None,
              download_if_missing=True,
              random_state=None, shuffle=False,
              categ_as_strings=False, return_X_y=False,
              as_frame=False):
    """Load the International Stroke Trial (IST) dataset.

    Download it if necessary.

    This is a randomized clinical trial of heparin and aspirin treatment
    for stroke patients.

    This dataset is derived from the corrected dataset available here:
    https://datashare.ed.ac.uk/handle/10283/128
    The webpage contains detailed descriptions.

    This version only includes pre-randomization variables, two main
    targets, and several additional targets related to side effects.

    The two main targets are:
    target_ID14 - death after 14 days
    target_OCCODE - outcome after 6 month.  Original study used
    ("dead" or "dependent") as outcome of interest

    Additionally there are 9 targets describing side effects at 14
    days: target_H14, target_ISC14, target_NK14, target_STRK14,
    target_HTI14, target_PE14, target_DVT14, target_TRAN14,
    target_NCB14

    **Variables**

    See https://datashare.ed.ac.uk/handle/10283/128 
    
    **Changes to the original dataset**

    - Only pretreatment variables, variables describing outcomes at 14
      days and 6 month outcome code are included
    - Change all N/Y variables to 0/1
    - Level H of RXHEP recoded as M for pilot study cases
    - Add var IS_PILOT indicating pilot study obtained by testing if
      RHEP24 is NaN.  The variable is only added if include_pilot is
      True.
    - RDATE variable has been split into RYEAR and RMONTH, month names
      have been translated to English
    - Recoded OCCODE to descriptive values, merge two "missing status"
      categories to "NA"

    Parameters
    ----------
    include_pilot : boolean, default=True
        Whether to include records from a pilot study with 984
        patients.  Some values (RATRIAL and RASP3) are missing in the
        pilot.
    
    include_location_vars : boolean, default=True
        Should variables describing hospitals and their locations be
        included. These are categorical variables with large number of
        levels.  The variables are: HOSPNUM, COUNTRY
    
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
    treatment_heparin_values = ["N", "L", "M"]
    hospnum_values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "93", "95", "96", "97", "98", "101", "102", "105", "106", "107", "108", "109", "110", "111", "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "125", "126", "127", "129", "130", "131", "132", "133", "134", "135", "136", "137", "139", "141", "142", "143", "144", "147", "149", "152", "153", "154", "155", "156", "158", "159", "161", "162", "163", "164", "165", "166", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "183", "184", "186", "187", "188", "189", "190", "191", "193", "194", "195", "196", "197", "198", "200", "201", "202", "203", "204", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "217", "218", "219", "220", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246", "247", "248", "249", "250", "251", "252", "253", "255", "256", "257", "258", "259", "260", "262", "264", "265", "267", "268", "271", "274", "278", "279", "281", "283", "285", "286", "289", "290", "291", "292", "293", "294", "295", "296", "297", "299", "300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "317", "319", "320", "322", "323", "324", "326", "327", "328", "330", "331", "332", "333", "334", "336", "337", "339", "341", "342", "343", "344", "345", "346", "348", "349", "350", "351", "352", "353", "354", "355", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "371", "372", "373", "374", "375", "376", "377", "378", "380", "381", "382", "383", "384", "387", "388", "390", "391", "392", "394", "395", "396", "399", "400", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "425", "428", "429", "430", "431", "433", "434", "435", "436", "437", "438", "439", "440", "441", "443", "445", "447", "449", "452", "453", "454", "455", "456", "457", "458", "461", "462", "463", "464", "465", "467", "468", "469", "470", "471", "472", "473", "474", "476", "477", "478", "479", "480", "481", "482", "483", "484", "485", "486", "487", "488", "491", "492", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "506", "507", "508", "510", "511", "512", "513", "514", "515", "516", "518", "519", "520", "521", "522", "523", "524", "527", "528", "529", "531", "532", "533", "534", "535", "536", "538", "539", "540", "541", "542", "543", "545", "546", "547", "548", "549", "550", "551", "553", "554", "557", "558", "559", "560", "561", "562", "563", "564", "565", "567", "568"]
    country_values = ['UK', 'ITAL', 'PORT', 'EIRE', 'BELG', 'FINL',
                      'AUSL', 'CZEC', 'USA', 'HUNG', 'NETH', 'NEW', 'SWIT', 'AUST',
                      'SLOV', 'SPAI', 'NORW', 'SWED', 'CHIL', 'GREE', 'POLA', 'TURK',
                      'SOUT', 'ISRA', 'SLOK', 'CANA', 'HONG', 'BRAS', 'INDI', 'ARGE',
                      'DENM', 'FRAN', 'SRI', 'ROMA', 'JAPA', 'SING']
    rconsc_values = ["F", "D", "U"]
    sex_values = ["F", "M"]
    ync_values = ["N", "Y", "C"] # C=Can't assess
    stype_values = ["PACS", "LACS", "TACS", "POCS", "OTH"]
    day_values = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    occode_values = ["dead", "dependent", "not_recovered", "recovered", "NA"]

    # attribute descriptions
    treatment_descr = [("treatment_asp", np.int32, "RXASP"),
                       ("treatment_hep", treatment_heparin_values, "RXHEP"),
                       ]
    target_descr = [("target_ID14", np.int32, "ID14"),
                    ("target_OCCODE", occode_values, "OCCODE"),
                    ("target_H14", np.int32, "H14"),
                    ("target_ISC14", np.int32, "ISC14"),
                    ("target_NK14", np.int32, "NK14"),
                    ("target_STRK14", np.int32, "STRK14"),
                    ("target_HTI14", np.int32, "HTI14"),
                    ("target_PE14", np.int32, "PE14"),
                    ("target_DVT14", np.int32, "DVT14"),
                    ("target_TRAN14", np.int32, "TRAN14"),
                    ("target_NCB14", np.int32, "NCB14"),
                    ]
    feature_descr = [("IS_PILOT", np.int32),
                     ("HOSPNUM", hospnum_values),
                     ("COUNTRY", country_values),
                     ("RDELAY", np.int32),
                     ("RCONSC", rconsc_values),
                     ("SEX", sex_values),
                     ("AGE", np.int32),
                     ("RSLEEP", np.int32),
                     ("RATRIAL", float),
                     ("RCT", np.int32),
                     ("RVISINF", np.int32),
                     ("RHEP24", float),
                     ("RASP3", float),
                     ("RSBP", np.int32),
                     ("RDEF1", ync_values),
                     ("RDEF2", ync_values),
                     ("RDEF3", ync_values),
                     ("RDEF4", ync_values),
                     ("RDEF5", ync_values),
                     ("RDEF6", ync_values),
                     ("RDEF7", ync_values),
                     ("RDEF8", ync_values),
                     ("STYPE", stype_values),
                     ("RYEAR", np.int32),
                     ("RMONTH", np.int32),
                     ("HOURLOCAL", np.int32),
                     ("MINLOCAL", np.int32),
                     ("DAYLOCAL", day_values),
                     ("EXPDD", float),
                     ("EXPD6", float),
                     ("EXPD14", float),
                     ]

    arch = ARCHIVE
    dataset_name = "IST"
    remove_vars = []
    if not include_location_vars:
        remove_vars += ["HOSPNUM", "COUNTRY"]
    if not include_prediction_model_vars:
        remove_vars += ["EXPDD", "EXPD6", "EXPD14"]
    if not include_pilot:
        remove_vars.append("IS_PILOT")
        record_mask = np.ones(19435, np.bool)
        record_mask[:984] = False
    else:
        record_mask = None
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
                            total_attrs=44, all_num=False,
                            remove_vars=remove_vars, record_mask=record_mask
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
