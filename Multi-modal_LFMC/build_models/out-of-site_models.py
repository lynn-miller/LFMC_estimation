""" LFMC Out-of-site Models
Creates the pool of out-of-site models.
""" 

import os
import numpy as np
import pandas as pd

import initialise
import common
from model_utils import reshape_data
from modelling_functions import create_models
from architecture_out_of_site import model_params


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
    # Model parameters settings
    # 
    # To find out more about any parameter, run `ModelParams().help('<parameter>')` 
    # =============================================================================
    model_params['modelName'] = 'out-of-site_models'
    model_params['description'] = 'Create a pool of out-of-site models'
    model_params['modisFilename'] = modis_csv
    model_params['prismFilename'] = prism_csv
    model_params['auxFilename'] = aux_csv
    model_params['splitMethod'] = 'bySite'
    model_params['splitFolds'] = 10
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
    # model_params['gpuMemory'] = 3800
    # model_params['maxWorkers'] = 5

    restart = False     # Change to True if retrying/restarting this script
    if not os.path.exists(model_params['modelDir']):
        os.makedirs(model_params['modelDir'])
    elif not restart:   # Don't over-write something important!
        raise FileExistsError(f"{model_params['modelDir']} exists but restart not requested")

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
    # - Runs (omitted for a single run)
    #   - Folds (for k-fold splitting)
    # 
    # =============================================================================
    X = {'modis': x_modis, 'prism': x_prism}
    with open(os.path.join(model_params['modelDir'], 'model_params.json'), 'w') as f:
        model_params.save(f)
    models = create_models(model_params, aux_data, X, y)

    print('\n\nResults Summary')
    print('===============\n')
    print(getattr(models, 'all_stats', None))

