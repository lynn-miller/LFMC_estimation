""" LFMC Out-of-site Models
Creates the pool of out-of-site models.
""" 

import os

import initialise
import common
from modelling_functions import create_models
from architecture_out_of_site import model_params
from display_utils import print_heading
import scenarios


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
    # Customize model parameters
    # 
    # To find out more about any parameter, run `ModelParams().help('<parameter>')` 
    # =============================================================================
    scenarios.out_of_site_scenario(model_params)
    model_params['modelName'] = 'out-of-site_models'
    model_params['description'] = 'Create a pool of out-of-site models'
    model_params['modelRuns'] = common.EVALUATION_RUNS
    model_params['modelDir'] = os.path.join(common.MODELS_DIR, model_params['modelName'])

    model_params['samplesFile'] = aux_csv
    model_params.add_input('optical', {'filename': modis_csv, 'channels': 7})
    model_params.add_input('weather', {'filename': prism_csv, 'channels': 7})
    
    # =============================================================================
    # Parameters for parallel execution on GPUs
    # =============================================================================
    # model_params['gpuDevice'] = 1
    # model_params['gpuMemory'] = 3800
    # model_params['maxWorkers'] = 5


    # =============================================================================
    # Builds and trains the LFMC models. 
    # =============================================================================
    models = create_models(model_params)
    print_heading('Results Summary', line_char='=', blank_before=2, blank_after=0)
    print_heading(model_params['modelName'], blank_before=1, blank_after=0)
    print(getattr(models, 'all_stats', None))
