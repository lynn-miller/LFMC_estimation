""" LFMC Within-site ablation test
Runs architecture ablation tests for the within-site models - each test
reverts an architecture change back to the Modis-tempCNN setting.
""" 

import os
import json
import numpy as np
import pandas as pd

import initialise
import common
from model_utils import reshape_data
from modelling_functions import run_experiment
from architecture_within_site import model_params


if __name__ == '__main__':
    # =============================================================================
    # Directories and Input files
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
    modis_csv = os.path.join(common.DATASETS_DIR, 'modis_365days.csv')
    prism_csv = os.path.join(common.DATASETS_DIR, 'prism_365days.csv')
    aux_csv = os.path.join(common.DATASETS_DIR, 'samples_365days.csv')

    # =============================================================================
    # Set up experiment parameters
    #
    # If the experiment dictionary contains a 'tests' key that is not
    # 'falsy' (False, None, 0, empty list) it is assumed to be a list of
    # tests to run. Each test will run with the specified model
    # parameters. Model parameters not specified will be the same for
    # each test, as set in the main model_params dictionary. A failed run
    # can be restarted by setting the 'restart' key to the test that
    # failed. This test and the remaining tests will then be run.
    # 
    # If 'tests' is 'falsy' then a single test will be run using the
    # parameters in the main model_params dictionary.
    # 
    # Other settings are:
    # - layerTypes: specifies which layers to include in the model
    # - Layer parameters should be specified as a list. The first entry
    #   in the list will be used for the first layer, etc.
    # - If the experiment includes changes to the layers, all non-default
    #   layer parameters need to be included. The parameters that are
    #   kept constant can be specified by including a key for the layer
    #   type in the experiment dictionary, and the value set to a
    #   dictionary of the constant parameters.
    # 
    # Model_parameters that cannot be changed in tests are:
    # - *Filename
    # - *Channels
    # - targetColumn
    # 
    # Example of setting layer parameters:  
    # {'name': 'Filters',
    #  'description': 'Test effect of conv layers filter sizes',
    #  'tests': [{'conv': {'filters': [32, 32, 32]}},
    #            {'conv': {'filters': [8, 8, 8]}},
    #            {'conv': {'filters': [32, 8, 8]}},
    #            {'conv': {'filters': [8, 32, 8]}},
    #            {'conv': {'filters': [8, 8, 32]}},
    #            {'conv': {'filters': [8, 16, 32]}},
    #            {'conv': {'filters': [32, 16, 8]}}],
    #  'conv': {'numLayers': 3, 'poolSize': [2, 3, 4]},
    #  'restart': 0}
    # =============================================================================
    experiment = {
        'name': 'within_site_ablation',
        'description': 'Within-site Architecture ablation tests',
        'layerTypes': ['modisConv', 'prismConv', 'fc'],
        'tests': [
            {'modisConv': {'numLayers': 5, 'filters': [32] * 5, 'poolSize': [0, 5, 2, 3, 4]},
             'prismConv': {'numLayers': 5, 'filters': [32] * 5, 'poolSize': [0, 5, 2, 3, 4]}
            },
            {'modisConv': {'numLayers': 3, 'filters': [8, 8, 8], 'poolSize': [2, 3, 4]},
             'prismConv': {'numLayers': 3, 'filters': [8, 8, 8], 'poolSize': [2, 3, 4]}
            },
            {'fc': {'numLayers': 2, 'units': [512, 512]}},
            {'fc': {'numLayers': 3, 'units': [256, 256, 256]}},
            {'dropoutRate': 0.5},
            {'batchSize': 32},
        ],
        'restart': None,
        'testNames': [
            'Conv filters: 32',
            'Conv layers: 3',
            'Dense layers: 2',
            'Dense units: 256',
            'Dropout: 0.5',
            'Batch size: 32',
        ]
    }

    # Save and display experiment details
    experiment_dir = os.path.join(common.MODELS_DIR, experiment['name'])
    restart = experiment.get('restart')
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    elif restart is None:
        raise FileExistsError(f'{experiment_dir} exists but restart not requested')
    experiment_file = f'experiment{restart}.json' if restart else 'experiment.json'
    with open(os.path.join(experiment_dir, experiment_file), 'w') as f:
        json.dump(experiment, f, indent=2)

    # =============================================================================
    # Model parameters settings
    # 
    # To find out more about any parameter, run `ModelParams().help('<parameter>')` 
    # =============================================================================
    model_params['modelName'] = experiment['name']
    model_params['description'] = experiment['description']
    model_params['modisFilename'] = modis_csv
    model_params['prismFilename'] = prism_csv
    model_params['auxFilename'] = aux_csv
    model_params['splitMethod'] = 'byYear'
    model_params['splitYear'] = 2014
    model_params['splitFolds'] = 4
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
    # Parameters for parallel execution on GPUs
    # =============================================================================
    # model_params['gpuDevice'] = 1
    # model_params['gpuMemory'] = 4096
    # model_params['maxWorkers'] = 4

    if not os.path.exists(model_params['modelDir']):
        os.makedirs(model_params['modelDir'])

    # =============================================================================
    # Prepare the data
    # =============================================================================
    modis_data = pd.read_csv(model_params['modisFilename'], index_col=0)
    x_modis = reshape_data(np.array(modis_data), model_params['modisChannels'])
    print(f'Modis shape: {x_modis.shape}')

    prism_data = pd.read_csv(model_params['prismFilename'], index_col=0)
    x_prism = reshape_data(np.array(prism_data), model_params['prismChannels'])
    print(f'Prism shape: {x_prism.shape}')

    aux_data = pd.read_csv(model_params['auxFilename'], index_col=0)
    y = aux_data[model_params['targetColumn']]

    # =============================================================================
    # Builds and trains the LFMC models. 
    # 
    # All models (if requested), predictions, and evaluation statistics
    # are saved to `model_dir`, with each run saved to a separate sub-directory
    # For each model created, predictions and evaluation statistics are also
    # returned as attributes of the model. These are stored as nested lists, the
    # structure is:
    # - Tests (if an experiment)
    #   - Runs (omitted for a single run)
    #     - Folds (for k-fold splitting)
    # =============================================================================
    X = {'modis': x_modis, 'prism': x_prism}
    models = run_experiment(experiment, model_params, aux_data, X, y)

    print('\n\nResults Summary')
    print('===============\n')
    for model in models:
        print(getattr(model, 'all_stats', None))
