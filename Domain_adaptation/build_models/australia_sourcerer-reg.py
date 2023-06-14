#!/usr/bin/env python
# coding: utf-8

""" LFMC Project Experiment
Script to run LFMC transfer learning tests
""" 

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import initialise
import common
from modelling_functions import run_experiment
from architecture_transfer import model_params
from model_parameters import ExperimentParams
from scenarios import australia_scenario
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
        'name': 'australia_sourcerer-reg',
        'description': 'Australia: pretrained on CONUS; Sourcerer - regulariser only; all training samples',
        'tests': [],
        'restart': None, 
        'rerun': None,
        'resumeAllTests': False,
    })
    
    folds_dir = os.path.join(common.MODELS_DIR, 'australia_gen-folds')
    pretrained_dir = os.path.join(common.MODELS_DIR, 'conus_base_models')
    
    seeds = [9013, 1815, 5313, 3945, 3632, 3875, 1782, 1393, 3708, 2914,
             4522, 3368, 6379, 3009, 3806, 6579, 4075, 1056, 5261, 4752]
    for n, s in enumerate(seeds):
        experiment['tests'].append({
            'testName': f'Ensemble {n+1}', 'randomSeed': s,
            'loadFolds': os.path.join(folds_dir, f'test{n}'),
            'pretrainedModel': os.path.join(pretrained_dir, f'test{n}')})

    # =============================================================================
    # Model parameters settings
    # 
    # To find out more about any parameter, run `ModelParams().help('<parameter>')` 
    # =============================================================================
    model_params['modelName'] = experiment['name']
    model_params['description'] = experiment['description']
    model_params['modelDir'] = os.path.join(common.MODELS_DIR, model_params['modelName'])
    australia_scenario(model_params)
#    model_params['modelRuns'] = 20
#    model_params['enableXla'] = False #True

    # Model inputs
    model_params['samplesFile'] = aux_csv
    model_params.add_input('optical', {'filename': modis_csv, 'channels': 7})
    model_params.add_input('weather', {'filename': era5_csv, 'channels': 7})
    
#    # Globe-LFMC Column Names
#    model_params['splitColumn'] = 'Group2'
#    model_params['yearColumn'] = 'Sampling year'

#    # Train/test split parameters
#    model_params['splitMethod'] = 'byValue'
#    model_params['splitFolds'] = 4
#    model_params['splitYear'] = 2014
#    model_params['yearFolds'] = 3
#    model_params['splitMax'] = True
#    model_params['saveFolds'] = True

    # Transfer learning parameters
#    model_params['pretrainedModel'] = os.path.join(common.MODELS_DIR, 'conus_base_models', 'test1')
    model_params['transferModel'] = {'method': 'sourcerer', 'freeze_bn': False, 'limit': True}
    model_params['commonNormalise'] = False

    # Other parameters
#    model_params['epochs'] = 1000
#    model_params['evaluateEpochs'] = 100
#    model_params['tempDir'] = common.TEMP_DIR
#    model_params['derivedModels'] = None
#    model_params['seedList'] = [
#        441, 780, 328, 718, 184, 372, 346, 363, 701, 358,
#        566, 451, 795, 237, 788, 185, 397, 530, 758, 633
#    ]
    
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
