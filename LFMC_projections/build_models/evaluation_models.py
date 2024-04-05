#!/usr/bin/env python
# coding: utf-8

""" LFMC Project Experiment
Script to run LFMC projections
""" 

import os

import initialise
import common
from modelling_functions import run_experiment
from architecture_projection import model_params
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
    
    modis_csv = os.path.join(common.DATASETS_DIR, 'modis_730days.csv')
    prism_csv = os.path.join(common.DATASETS_DIR, 'prism_730days.csv')
    aux_csv = os.path.join(common.DATASETS_DIR, 'samples_730days.csv')

    # =============================================================================
    # Experiment parameters settings
    # 
    # To find out more about any parameter, run `ExperimentParams().help('<parameter>')` 
    # =============================================================================
    experiment = ExperimentParams({
        'name': 'evaluation_models',
        'description': 'Projection: Lead times from 0 to 1 year',
        'blocks': {'fc': {}},
        'tests': [
            {'testName': 'Nowcasting',
             'inputs': {'optical': {'start': -365, 'end': 0},
                        'weather': {'start': -365, 'end': 0}}},
            {'testName': '1-month lead time',
             'inputs': {'optical': {'start': -336, 'end': 29},
                        'weather': {'start': -336, 'end': 29}}},
            {'testName': '2-months lead time',
             'inputs': {'optical': {'start': -305, 'end': 60},
                        'weather': {'start': -305, 'end': 60}}},
            {'testName': '3-months lead time',
             'inputs': {'optical': {'start': -275, 'end': 90},
                        'weather': {'start': -275, 'end': 90}}},
            {'testName': '4-months lead time',
             'inputs': {'optical': {'start': -244, 'end': 121},
                        'weather': {'start': -244, 'end': 121}}},
            {'testName': '5-months lead time',
             'inputs': {'optical': {'start': -214, 'end': 151},
                        'weather': {'start': -214, 'end': 151}}},
            {'testName': '6-months lead time',
             'inputs': {'optical': {'start': -183, 'end': 182},
                        'weather': {'start': -183, 'end': 182}}},
            {'testName': '7-months lead time',
             'inputs': {'optical': {'start': -153, 'end': 212},
                        'weather': {'start': -153, 'end': 212}}},
            {'testName': '8-months lead time',
             'inputs': {'optical': {'start': -123, 'end': 242},
                        'weather': {'start': -123, 'end': 242}}},
            {'testName': '9-months lead time',
             'inputs': {'optical': {'start': -92, 'end': 273},
                        'weather': {'start': -92, 'end': 273}}},
            {'testName': '10-months lead time',
             'inputs': {'optical': {'start': -62, 'end': 303},
                        'weather': {'start': -62, 'end': 303}}},
            {'testName': '11-months lead time',
             'inputs': {'optical': {'start': -31, 'end': 334},
                        'weather': {'start': -31, 'end': 334}}},
            {'testName': '1-year lead time',
             'inputs': {'optical': {'start': -1, 'end': 364},
                        'weather': {'start': -1, 'end': 364}}},
        ],
        'restart': 0, 
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
    model_params.add_input('weather', {'filename': prism_csv, 'channels': 7})
    
    # Globe-LFMC Column Names
    model_params['splitColumn'] = 'Site'
    model_params['splitStratify'] = 'Land Cover'
    model_params['yearColumn'] = 'Sampling year'

    # Train/test split parameters
    model_params['splitMethod'] = 'bySite'
    model_params['splitFolds'] = 4
    model_params['splitYear'] = 2014
    model_params['yearFolds'] = 4

    model_params['modelRuns'] = common.EVALUATION_RUNS
    model_params['tempDir'] = common.TEMP_DIR
    model_params['modelDir'] = os.path.join(common.MODELS_DIR, model_params['modelName'])
    model_params['derivedModels'] = common.DERIVED_MODELS
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
    model_params['maxWorkers'] = 8       # Number of workers (parallel processes)
    model_params['gpuList'] = [0, 1]     # List of GPUs to use
    model_params['gpuMemory'] = 4800     # GPU memory for each worker

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
    print_heading('Results Summary', line_char='=', blank_before=2, blank_after=0)
    for num, model in enumerate(models):
        test = experiment['tests'][num]
        try:
            test_name = test.get('testName', None) or experiment['testNames'][num]
        except:
            test_name = '<unnamed test>'
        print_heading(f'Test {num}: {test_name}', blank_before=1, blank_after=0)
        print(getattr(model, 'all_stats', None))