#!/usr/bin/env python
# coding: utf-8

""" LFMC Project Experiment
Script to run LFMC transfer learning tests
""" 

import os

import initialise
import common
from modelling_functions import run_experiment
from architecture_transfer import model_params
from model_parameters import ExperimentParams
from display_utils import print_heading


if __name__ == '__main__':
    # =============================================================================
    # Input files
    #
    # Change these settings as required
    # - `modis_csv`: The file containing extracted MODIS data for each
    #   sample, created by `Extract MODIS Data.ipynb`
    # - `prism_csv`: The file containing extracted PRISM data for each
    #   sample, created by `Extract PRISM Data.ipynb`
    # - `aux_csv`: The file containing extracted sample labels, DEM, climate zone
    #   and other auxiliary data, created by `Extract Auxiliary Data.ipynb`.
    #     
    # =============================================================================
    
    modis_csv = os.path.join(common.DATASETS_DIR, 'australia_modis_365days.csv')
    era5_csv = os.path.join(common.DATASETS_DIR, 'australia_era5_365days.csv')
    aux_csv = os.path.join(common.DATASETS_DIR, 'australia_samples_365days.csv')

    # =============================================================================
    # Experiment parameters settings
    # 
    # To find out more about any parameter, run `ExperimentParams().help('<parameter>')` 
    # =============================================================================
    experiment = ExperimentParams({
        'name': 'australia_gen-folds',
        'description': 'Generate folds for Australian model testing',
        'tests': [
            {'testName': 'Ensemble 1', 'randomSeed': 9013},
            {'testName': 'Ensemble 2', 'randomSeed': 1815},
            {'testName': 'Ensemble 3', 'randomSeed': 5313},
            {'testName': 'Ensemble 4', 'randomSeed': 3945},
            {'testName': 'Ensemble 5', 'randomSeed': 3632},
            {'testName': 'Ensemble 6', 'randomSeed': 3875},
            {'testName': 'Ensemble 7', 'randomSeed': 1782},
            {'testName': 'Ensemble 8', 'randomSeed': 1393},
            {'testName': 'Ensemble 9', 'randomSeed': 3708},
            {'testName': 'Ensemble 10', 'randomSeed': 2914},
            {'testName': 'Ensemble 11', 'randomSeed': 4522},
            {'testName': 'Ensemble 12', 'randomSeed': 3368},
            {'testName': 'Ensemble 13', 'randomSeed': 6379},
            {'testName': 'Ensemble 14', 'randomSeed': 3009},
            {'testName': 'Ensemble 15', 'randomSeed': 3806},
            {'testName': 'Ensemble 16', 'randomSeed': 6579},
            {'testName': 'Ensemble 17', 'randomSeed': 4075},
            {'testName': 'Ensemble 18', 'randomSeed': 1056},
            {'testName': 'Ensemble 19', 'randomSeed': 5261},
            {'testName': 'Ensemble 20', 'randomSeed': 4752},
            {'testName': 'Ensemble 21', 'randomSeed': 7338},
            {'testName': 'Ensemble 22', 'randomSeed': 3455},
            {'testName': 'Ensemble 23', 'randomSeed': 4447},
            {'testName': 'Ensemble 24', 'randomSeed': 1281},
            {'testName': 'Ensemble 25', 'randomSeed': 8873},
            {'testName': 'Ensemble 26', 'randomSeed': 9600},
            {'testName': 'Ensemble 27', 'randomSeed': 4585},
            {'testName': 'Ensemble 28', 'randomSeed': 8270},
            {'testName': 'Ensemble 29', 'randomSeed': 4415},
            {'testName': 'Ensemble 30', 'randomSeed': 9810},
            {'testName': 'Ensemble 31', 'randomSeed': 6638},
            {'testName': 'Ensemble 32', 'randomSeed': 2742},
            {'testName': 'Ensemble 33', 'randomSeed': 9732},
            {'testName': 'Ensemble 34', 'randomSeed': 2073},
            {'testName': 'Ensemble 35', 'randomSeed': 4349},
            {'testName': 'Ensemble 36', 'randomSeed': 9011},
            {'testName': 'Ensemble 37', 'randomSeed': 8482},
            {'testName': 'Ensemble 38', 'randomSeed': 8209},
            {'testName': 'Ensemble 39', 'randomSeed': 5133},
            {'testName': 'Ensemble 40', 'randomSeed': 9220},
            {'testName': 'Ensemble 41', 'randomSeed': 4401},
            {'testName': 'Ensemble 42', 'randomSeed': 6584},
            {'testName': 'Ensemble 43', 'randomSeed': 9566},
            {'testName': 'Ensemble 44', 'randomSeed': 8028},
            {'testName': 'Ensemble 45', 'randomSeed': 8149},
            {'testName': 'Ensemble 46', 'randomSeed': 7036},
            {'testName': 'Ensemble 47', 'randomSeed': 1245},
            {'testName': 'Ensemble 48', 'randomSeed': 7460},
            {'testName': 'Ensemble 49', 'randomSeed': 3239},
            {'testName': 'Ensemble 50', 'randomSeed': 1888},
        ],
        'restart': None, 
        'rerun': None,
        'resumeAllTests': False,
    })

    # =============================================================================
    # Model parameters settings
    # 
    # To find out more about any parameter, run `ModelParams().help('<parameter>')` 
    # =============================================================================
    model_params['modelName'] = experiment['name']
    model_params['description'] = experiment['description']

    # Model inputs
    model_params['samplesFile'] = aux_csv
    model_params.add_input('optical', {'filename': modis_csv, 'channels': 7})
    model_params.add_input('weather', {'filename': era5_csv, 'channels': 7})
    
    # Globe-LFMC Column Names
    model_params['splitColumn'] = 'Group2'
    model_params['yearColumn'] = 'Sampling year'

    # Train/test split parameters
    model_params['splitMethod'] = 'byValue'
    model_params['splitFolds'] = 4
    model_params['splitYear'] = 2014
    model_params['yearFolds'] = 3
    model_params['splitMax'] = True
    model_params['saveFolds'] = True

    model_params['epochs'] = 0
    model_params['auxAugment'] = True
    model_params['modelRuns'] = -1
    model_params['tempDir'] = common.TEMP_DIR
    model_params['modelDir'] = os.path.join(common.MODELS_DIR, model_params['modelName'])
    model_params['derivedModels'] = None
    model_params['seedList'] = [
        441, 780, 328, 718, 184, 372, 346, 363, 701, 358,
        566, 451, 795, 237, 788, 185, 397, 530, 758, 633,
        632, 941, 641, 519, 162, 215, 578, 919, 917, 585,
        914, 326, 334, 366, 336, 413, 111, 599, 416, 230,
        191, 700, 697, 332, 910, 331, 771, 539, 575, 457
    ]
    
    # =============================================================================
    # GPU and parallelisation parameters
    #
    # Change these settings as required
    #
    # Note: the gpuList parameter is either a list of GPU numbers or a list of
    # lists of GPU numbers. Each worker is assigned a GPU (or GPUs) from this list
    # on a round-robin basis. If the list has fewer entries than the number of
    # workers, it will be re-cycled until GPUs have been assigned to all workers.
    # GPU numbers are indexes of tf.config.list_physical_devices('GPU')
    # =============================================================================
    model_params['maxWorkers'] = 12     # Number of workers (parallel processes)
    model_params['gpuList'] = [0, 1]    # List of GPUs to use
    model_params['gpuMemory'] = 256     # GPU memory for each worker

    # =============================================================================
    # Builds and trains the LFMC models. After training the models, several
    # derived models are created and evaluated.
    # 
    # All models (if requested), predictions, and evaluation statistics
    # are saved to `model_dir`, with each run saved to a separate sub-directory
    # For each model created, predictions and evaluation statistics are also
    # returned as attributes of the model. These are stored as nested lists, the
    # structure is:
    # - Runs (omitted for a single run)
    #   - Folds (for k-fold splitting)
    # 
    # =============================================================================
    models = run_experiment(experiment, model_params)
