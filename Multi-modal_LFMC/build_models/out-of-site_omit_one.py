""" LFMC Out-of-site omit one input
Runs input ablation tests for the out-of-site models - each test omits
a new input or a group of auxiliary inputs.
""" 

import os

import initialise
import common
from modelling_functions import run_experiment
from architecture_out_of_site import model_params
from model_parameters import ExperimentParams
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
    # Set up experiment parameters
    #
    # If the experiment dictionary contains a 'tests' key that is not
    # 'falsy' (False, None, 0, empty list) it is assumed to be a list of
    # tests to run. Each test will run with the specified model
    # parameters. Model parameters not specified will be the same for
    # each test, as set in the main model_params dictionary. A failed run
    # can be restarted by setting the 'restart' key to the test that
    # failed. This test and the remaining tests will then be run.
    # =============================================================================
    experiment = ExperimentParams({
        'name': 'out-of-site_omit_one',
        'description': 'Input abalation tests',
        'tests': [
            {'testName': 'Without weather',
             'dataSources': ['optical', 'aux'],
             'inputs': {'weather': None},
             'blocks': {'weatherConv': None }},
            {'testName': 'Without climate zone', 'auxOneHotCols': []},
            {'testName': 'With day-of-year',
             'auxColumns': ['Day_sin', 'Day_cos', 'Elevation', 'Slope', 'Aspect_sin',
                            'Aspect_cos', 'Long_sin', 'Long_cos', 'Lat_norm']},
            {'testName': 'Without topography',
             'auxColumns': ['Long_sin', 'Long_cos', 'Lat_norm']},
            {'testName': 'Without location',
             'auxColumns': ['Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos']},
        ],
        'restart': None,
    })

    # =============================================================================
    # Model parameters settings
    # 
    # To find out more about any parameter, run `ModelParams().help('<parameter>')` 
    # =============================================================================
    scenarios.out_of_site_scenario(model_params)
    model_params['modelName'] = experiment['name']
    model_params['description'] = experiment['desciption']
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