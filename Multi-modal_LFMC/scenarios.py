""" Tranfer Learning Scenarios
Defines the scenarios used to evaluate transfer learning methods
"""

import initialise
import common


def _set_common_params(model_params):
    model_params['saveFolds'] = True
    model_params['enableXla'] = False
    model_params['tempDir'] = common.TEMP_DIR
    model_params['seedList'] = [
        441, 780, 328, 718, 184, 372, 346, 363, 701, 358,
        566, 451, 795, 237, 788, 185, 397, 530, 758, 633,
        632, 941, 641, 519, 162, 215, 578, 919, 917, 585,
        914, 326, 334, 366, 336, 413, 111, 599, 416, 230,
        191, 700, 697, 332, 910, 331, 771, 539, 575, 457,
    ]

def common_params(model_params):
    _set_common_params(model_params)
    

def out_of_site_scenario(model_params, set_common=True):
    # Common parameters
    if set_common:
        _set_common_params(model_params)

    # Train/test split parameters
    model_params['splitMethod'] = 'byValue'
    model_params['splitStratify'] = 'Land Cover'
    model_params['splitColumn'] = 'Site'
    model_params['splitFolds'] = 10


def within_site_scenario(model_params, set_common=True):
    # Common parameters
    if set_common:
        _set_common_params(model_params)

    # Train/test split parameters
    model_params['splitMethod'] = 'byYear'
    model_params['yearColumn'] = 'Sampling year'
    model_params['splitFolds'] = None
    model_params['splitYear'] = 2014
    model_params['yearFolds'] = 4
