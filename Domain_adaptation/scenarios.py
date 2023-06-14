""" Tranfer Learning Scenarios
Defines the scenarios used to evaluate transfer learning methods
"""

import initialise
import common


def _set_common_params(model_params):
    model_params['yearColumn'] = 'Sampling year'
    model_params['splitYear'] = 2014
    model_params['splitMax'] = True
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
    

def australia_scenario(model_params):
    # Common parameters
    _set_common_params(model_params)

    # Train/test split parameters
    model_params['splitMethod'] = 'byValue'
    model_params['splitColumn'] = 'Group2'
    model_params['splitFolds'] = 4
    model_params['yearFolds'] = 3

    # Epochs
    if model_params['targetColumn'] == 'LFMC value':
        model_params['epochs'] = 1000
        model_params['evaluateEpochs'] = 100
    elif model_params['targetColumn'] == 'Czone3':
        model_params['epochs'] = 500
        model_params['evaluateEpochs'] = 50
    elif model_params['targetColumn'] == 'LC Category':
        model_params['epochs'] = 100
        model_params['evaluateEpochs'] = 10
    

def conus_scenario(model_params):
    # Common parameters
    _set_common_params(model_params)

    # Train/test split parameters
    model_params['splitMethod'] = 'byYear'
    model_params['splitFolds'] = None
    model_params['yearFolds'] = 4

    # Other parameters
    model_params['epochs'] = 50
    model_params['modelRuns'] = 50
    model_params['saveModels'] = True


def europe_scenario(model_params):
    # Common parameters
    _set_common_params(model_params)

    # Train/test split parameters
    model_params['splitMethod'] = 'byValue'
    model_params['splitColumn'] = 'Group2'
    model_params['splitFolds'] = 4
    model_params['yearFolds'] = 3

    # Epochs
    if model_params['targetColumn'] == 'LFMC value':
        model_params['epochs'] = 500
        model_params['evaluateEpochs'] = 50
    elif model_params['targetColumn'] == 'Czone3':
        model_params['epochs'] = 200
        model_params['evaluateEpochs'] = 20
    elif model_params['targetColumn'] == 'LC Category':
        model_params['epochs'] = 200
        model_params['evaluateEpochs'] = 20

