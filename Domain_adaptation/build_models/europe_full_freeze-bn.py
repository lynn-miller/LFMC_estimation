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
    
    modis_csv = os.path.join(common.DATASETS_DIR, 'europe_modis_365days.csv')
    era5_csv = os.path.join(common.DATASETS_DIR, 'europe_era5_365days.csv')
    aux_csv = os.path.join(common.DATASETS_DIR, 'europe_samples_365days.csv')

    # =============================================================================
    # Experiment parameters settings
    # 
    # To find out more about any parameter, run `ExperimentParams().help('<parameter>')` 
    # =============================================================================
    folds_dir = os.path.join(common.MODELS_DIR, 'europe_gen-folds')
    pretrained_dir = os.path.join(common.MODELS_DIR, 'conus_base_models')
    experiment = ExperimentParams({
        'name': 'europe_full_freeze-bn',
        'description': 'Europe: Pretrained on CONUS; Freeze BN layers; all training data',
        'tests': [
            {'testName': 'Ensemble 1', 'randomSeed': 9013,
             'loadFolds': os.path.join(folds_dir, 'test0'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test0')},
            {'testName': 'Ensemble 2', 'randomSeed': 1815,
             'loadFolds': os.path.join(folds_dir, 'test1'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test1')},
            {'testName': 'Ensemble 3', 'randomSeed': 5313,
             'loadFolds': os.path.join(folds_dir, 'test2'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test2')},
            {'testName': 'Ensemble 4', 'randomSeed': 3945,
             'loadFolds': os.path.join(folds_dir, 'test3'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test3')},
            {'testName': 'Ensemble 5', 'randomSeed': 3632,
             'loadFolds': os.path.join(folds_dir, 'test4'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test4')},
            {'testName': 'Ensemble 6', 'randomSeed': 3875,
             'loadFolds': os.path.join(folds_dir, 'test5'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test5')},
            {'testName': 'Ensemble 7', 'randomSeed': 1782,
             'loadFolds': os.path.join(folds_dir, 'test6'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test6')},
            {'testName': 'Ensemble 8', 'randomSeed': 1393,
             'loadFolds': os.path.join(folds_dir, 'test7'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test7')},
            {'testName': 'Ensemble 9', 'randomSeed': 3708,
             'loadFolds': os.path.join(folds_dir, 'test8'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test8')},
            {'testName': 'Ensemble 10', 'randomSeed': 2914,
             'loadFolds': os.path.join(folds_dir, 'test9'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test9')},
            {'testName': 'Ensemble 11', 'randomSeed': 4522,
             'loadFolds': os.path.join(folds_dir, 'test10'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test10')},
            {'testName': 'Ensemble 12', 'randomSeed': 3368,
             'loadFolds': os.path.join(folds_dir, 'test11'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test11')},
            {'testName': 'Ensemble 13', 'randomSeed': 6379,
             'loadFolds': os.path.join(folds_dir, 'test12'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test12')},
            {'testName': 'Ensemble 14', 'randomSeed': 3009,
             'loadFolds': os.path.join(folds_dir, 'test13'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test13')},
            {'testName': 'Ensemble 15', 'randomSeed': 3806,
             'loadFolds': os.path.join(folds_dir, 'test14'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test14')},
            {'testName': 'Ensemble 16', 'randomSeed': 6579,
             'loadFolds': os.path.join(folds_dir, 'test15'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test15')},
            {'testName': 'Ensemble 17', 'randomSeed': 4075,
             'loadFolds': os.path.join(folds_dir, 'test16'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test16')},
            {'testName': 'Ensemble 18', 'randomSeed': 1056,
             'loadFolds': os.path.join(folds_dir, 'test17'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test17')},
            {'testName': 'Ensemble 19', 'randomSeed': 5261,
             'loadFolds': os.path.join(folds_dir, 'test18'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test18')},
            {'testName': 'Ensemble 20', 'randomSeed': 4752,
             'loadFolds': os.path.join(folds_dir, 'test19'),
             'pretrainedModel': os.path.join(pretrained_dir, 'test19')},
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
    model_params['modelRuns'] = 20
    model_params['enableXla'] = False #True

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

    # Transfer learning parameters
    model_params['pretrainedModel'] = os.path.join(common.MODELS_DIR, 'conus_base_models', 'test1')
    model_params['transferModel'] = {'method': 'freeze_layers', 'layers': ['bnorm']}
    model_params['commonNormalise'] = False

    # Other parameters
    model_params['epochs'] = 500
    model_params['evaluateEpochs'] = 100
    model_params['tempDir'] = common.TEMP_DIR
    model_params['modelDir'] = os.path.join(common.MODELS_DIR, model_params['modelName'])
    model_params['derivedModels'] = None
    model_params['seedList'] = [
        441, 780, 328, 718, 184, 372, 346, 363, 701, 358,
        566, 451, 795, 237, 788, 185, 397, 530, 758, 633
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
    model_params['maxWorkers'] =  10    # Number of workers (parallel processes)
    model_params['gpuList'] =  [0, 1]   # List of GPUs to use
    model_params['gpuMemory'] = 3000    # GPU memory for each worker

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
