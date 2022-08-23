"""Functions to build and evaluate LFMC models"""

import contextlib
import numpy as np
import os
import pandas as pd
import warnings

from copy import deepcopy
from multiprocess import get_context, Manager
from time import sleep

from model_list import ModelList
from model_parameters import ModelParams
from data_splitting_utils import train_test_split, kfold_split
from data_prep_utils import reshape_data, create_onehot_enc
from model_utils import gen_test_results, create_ensembles, merge_kfold_results
from model_utils import train_test_model
from display_utils import print_heading


train_test_model.gpu = None


def _train_test_subprocess(pool, **kwargs):
    """Creates a sub-process to train and test a model.
    
    Calls ``train_test_model`` as a sub-process using a multiprocess
    worker pool. This ensures the GPU memory is released after
    completing each model run.

    Parameters
    ----------
    pool : Pool
        A pool of multiprocess workers
    **kwargs : keyword arguments
        The arguments that should be passed to ``train_test_model``.
        Must be ``model_params``, ``train``, ``val``, ``test``.

    Returns
    -------
    results : AsyncResult
        The object that will contain the model results once the run has
        finished. The ``get()`` method will return an object of type
        determined by ``model_params['modelClass']`` and defaults to
        ``LfmcModel``.
    """
    def success(model):
        name = model.params['modelName']
        min_loss = model.train_result['minLoss']
        run_time = model.train_result['runTime']
        if model.params['diagnostics']:
            print(f"{name} training results: minLoss: {min_loss:.3f}, runTime: {run_time:.3f}")

    results = pool.apply_async(train_test_model, kwds=kwargs, callback=success)
    sleep(2)    # Wait a little so sub-processes aren't submitted too quickly
    return results


def _read_data(model_params, samples=None, X={}, inputs_needed=True):
    def read_files(source, filename, multi_samples, inputs_needed=True):
        if isinstance(filename, list):    # Multiple file for this source
            sep = '\n    '
            print(f"Reading {source} files: {sep}{sep.join(filename)}")
            data = [pd.read_csv(fn, index_col=0) for fn in filename]
            if multi_samples:             # Extend the number of samples
                if len(filename) == len(multi_samples):
                    for df1 in data:
                        df1.columns = data[0].columns
                    data = pd.concat(data, keys=multi_samples)
                else:
                    raise Exception(f'Incorrect number of {source} files: '
                                    f'{len(filename)} found; {len(multi_samples)} expected.')
            else:                         # Extend the time-series
                data = pd.concat(data, axis=1)
        elif filename:                    # Single file for this source
            print(f"Reading {source} file {filename}")
            data = pd.read_csv(filename, index_col=0)
            if multi_samples:             # Replicate the file once for each key
                data = pd.concat([data] * len(multi_samples), keys=multi_samples)
        elif inputs_needed:               # Input required but not provided
            raise Exception(f'No {source} file provided')
        else:                             # Input not provided and not required
            data = None
        return data
        
    multi_samples = model_params.get('multiSamples', [])
    if samples is None:           # No samples yet, so load from source file
        filename = model_params.get('samplesFile', None)
        samples = read_files('samples', filename, multi_samples, inputs_needed)
    if model_params['samplesFilter']:
        filter_column = model_params['samplesFilter'][0]
        filter_method = model_params['samplesFilter'][1]
        filter_params = model_params['samplesFilter'][2:]
        filter_str = ', '.join([str(x) for x in filter_params])
        print(f"Samples filtered using \"{filter_column}.{filter_method}({filter_str})\"")
        filter_ = getattr(samples[filter_column], filter_method)(*filter_params)
        samples = samples[filter_]

    X = X or {}
    temp_X = {}
    for input_name in model_params['dataSources']:
        if input_name == 'aux':
            continue                    # Auxiliaries extracted from samples so nothing to do here
        elif input_name in X.keys():    # Data for this input has been provided
            temp_X[input_name] = X[input_name]
        else:                           # No data for this input yet, so load it from source files
            filename = model_params['inputs'][input_name].get('filename', None)
            data = read_files(input_name, filename, multi_samples, inputs_needed)
            if data is not None:        # No data for this input yet. It will be loaded later.
                temp_X[input_name] = data

    return samples, temp_X

    
def _set_run_params(model_params, run):
    run_params = ModelParams(deepcopy(model_params))
    run_params['modelName'] = '_'.join([run_params['modelName'], f"run{run}"])
    run_params['modelDir'] = os.path.join(run_params['modelDir'], f"run{run}")
    run_params['run'] = run
    model_seeds = model_params['seedList']
    if model_seeds is None or len(model_seeds) == 0:
        model_seeds = [model_params['modelSeed']]
    run_params['modelSeed'] = model_seeds[run] if run < len(model_seeds) else None
    if model_params['resplit']:
        if model_params['seedList']:
            run_params['randomSeed'] = run_params['modelSeed']
        elif run > 0:
            run_params['randomSeed'] = None
    return run_params


def _set_test_params(experiment, model_params, test_num):
    test = deepcopy(experiment['tests'][test_num])
    try:
        test_name = test.pop('testName', None) or experiment['testNames'][test_num]
        print(f'Test {test_num}: {test_name} - {test}\n')
    except:
        test_name = f"Test{test_num}"
        print(f'Test {test_num} - {test}\n')
    test_params = ModelParams(deepcopy(model_params))
    test_params['testName'] = test_name
    test_params['modelName'] = '_'.join([test_params['modelName'], f"test{test_num}"])
    test_params['modelDir'] = os.path.join(test_params['modelDir'], f"test{test_num}")
    inputs = test.pop('inputs', {})
    test_params.update(test)
    # Update model inputs
    for input_name, input_params in inputs.items():
        test_params['inputs'][input_name].update(input_params)
    # Update model layers
    for block in experiment['blocks'].keys() & test.keys():
        layer_parms = deepcopy(experiment['blocks'][block])
        layer_parms.update(test[block])
        test_params.set_layers(
            **{block.replace('Conv', '') + '_layers': layer_parms['numLayers']})
        for key, values in layer_parms.items():
            if key != 'numLayers':
                is_list = isinstance(values, list)
                for layer_num, layer in enumerate(test_params[block]):
                    layer[key] = values[layer_num] if is_list else values
    # Can use either splitFolds or yearFolds with byYear method
    if (test_params['splitMethod'] == 'byYear') and (test_params['yearFolds'] is None):
        test_params['yearFolds'] = test_params['splitFolds']
    # Get any sources that need re-loading
    reload = [src for src, f in inputs.items() if 'filename' in f.keys()]
    return test_params, reload


def run_kfold_model(pool, model_params, samples, X, y, parent_results):
    """Runs a K-fold model.
    
    Splits the data into folds, and builds and evaluates a model for
    each fold.

    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters.
    samples : DataFrame
        A data frame containing the samples.
    X : dict
        The predictor data. The keys are the name of each source. Each
        value is an array of the predictors for that source. The first
        dimension is assumed to correspond to the rows of the samples
        data frame in length and order.
    y : Array
        The labels. A 1-dimensional array containing the label for each
        sample.

    Returns
    -------
    models : ModelList
        A list (length k) of the k-fold models. To save memory, the
        Keras components are removed from the models before they are
        returned.
    """

    model_name = model_params['modelName']
    model_dir = os.path.join(model_params['modelDir'], '')
    model_params.save('model_params.json')
    models = ModelList(model_name, model_dir)

    folds = []
    for fold, data in kfold_split(model_params, samples, X, y, parent_results):
        fold_name = f'fold_{fold}'
        folds.append(fold_name)
        fold_params = ModelParams(deepcopy(model_params))
        fold_params['modelName'] = f"{model_name}_{fold_name}"
        fold_params['modelDir'] = os.path.join(model_dir, fold_name)
        fold_params['fold'] = fold
#        normalise_data(data, fold_params)
#        print(f"Inputs: {fold_params['inputs']}, Target normalisation: {fold_params['targetNormalise']}")
        model = _train_test_subprocess(pool, model_params=fold_params, **data)
        models.append(model)
        model_params['plotModel'] = False

    for num, model in enumerate(models):
        models[num] = model.get()
         
    merge_kfold_results(
        model_dir, models, folds=folds, epochs=model_params['evaluateEpochs'])
        
    return models


def prepare_labels(model_params, samples):
    target = model_params['targetColumn']
    if samples is not None and (target in samples):
        print(f"Setting target to \"{target}\"", end='')
        y = samples[target]   # Extract labels from the target column
        if model_params['targetTransform']:
            xform_method = model_params['targetTransform'][0]
            xform_params = model_params['targetTransform'][1:]
            xform_str = ', '.join([str(x) for x in xform_params])
            print(f", transformed using \"{xform_method}({xform_str})\"")
            y = getattr(y, xform_method)(*xform_params)
        else:
            print('')
        if model_params.get('classify', False):
            model_params['numClasses'] = y.nunique()
    else:       # Labels not provided
        raise Exception('No target data provided')
    return y

    
def prepare_data(model_params, samples=None, X={}, predict=False):
    """Prepares data for LFMC model training or prediction.
    
    Prepares the data for an LFMC model test.  
     - Only the inputs in the ``dataSources`` model parameter are
       processed
     - Extracts the auxiliary columns from the sample data
     - One-hot encodes any categorical auxiliary columns
     - Extracts a sub-series from each time series input based on the
       start and end model parameters.
     - Normalises the timeseries data using ``model_params`` settings
     - Augments the auxiliary data with the last item in each time
       series if required

    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters.
    samples : DataFrame or None, optional
        A data frame containing the samples. If None, the samples will
        be loaded from the file specified in the ``samplesFile`` model
        parameter. The default is None
    X : dict or None, optional
        The predictor data. The keys are the name of each source. Each
        value is an array of the predictors for that source. The first
        dimension is assumed to correspond to the rows of the samples
        data frame in length and order. Any keys not in ``dataSources``
        are ignored, as is any ``aux`` key (``aux`` is generated from
        the ``samples``). If a key for any of the ``dataSources`` is
        missing, that source is loaded from the file specified in the
        ``<source>Filename`` model parameter. If X is ``None`` or an
        empty dictionary then all ``dataSources`` sources are loaded
        from the source files. The default is {}.

    Returns
    -------
    X : dict
        A dictionary of the prepared data. Dictionary keys are the input
        names and values are numpy arrays of each input.

    """
    
    def prepare_aux_data(model_params, samples, predict):
        if isinstance(model_params['auxColumns'], int):
            aux_start = len(samples.columns) - model_params['auxColumns']
            x_aux = np.array(samples.iloc[:, aux_start:])
            print('Auxiliary columns:', list(samples.iloc[:, aux_start:].columns))
        else:
            x_aux = np.array(samples[model_params['auxColumns']])
            print('Auxiliary columns:', model_params['auxColumns'])
        onehot_cols = model_params['auxOneHotCols']
        if onehot_cols:
            print('One-hot encoded columns:', model_params['auxOneHotCols'])
        if onehot_cols:
            if predict:
                fit_data = None
            else:
                fit_data = samples[onehot_cols]
            onehot_enc = create_onehot_enc(onehot_cols,
                                           fit_data,
                                           model_dir=model_params['modelDir'],
                                           save=model_params.get('saveModels', False))
            onehot_data = onehot_enc.transform(samples[onehot_cols].fillna('').to_numpy())
            x_aux = np.concatenate([x_aux, onehot_data], axis=1)
        return x_aux
    
    def prepare_ts_data(model_params, input_params, input_data, predict):
        input_data = reshape_data(np.array(input_data), input_params['channels'])
        print(f'{input_name.capitalize()} input shape: {input_data.shape}')
        include_channels = input_params.get('includeChannels', [])
        if include_channels:
            input_data = input_data[:, :, include_channels]
        start = input_params['start'] or 0
        if input_params['end']:
            end = -input_params['end']
        else:
            end = None
        input_data = input_data[:, -start:end, :]
        if model_params['diagnostics']:
            print(f'Start: {start}; End: {end}')
            for x in range(0, input_data.shape[1], 24):
                print(input_data[0, x:x+24, 0])
        return input_data
        # normal = input_params['normalise']
        # return normalise(input_data, **normal,
        #                  input_name=input_name,
        #                  model_dir=model_params['modelDir'],
        #                  save_params=model_params['saveModels'],
        #                  load_params=predict)

    if 'aux' in model_params['dataSources']:
        augment = model_params['auxAugment']
        temp_aux = prepare_aux_data(model_params, samples, predict)
    else:
        temp_aux = None   # No auxiliary data
        augment = False   # Ignore the auxAugment setting if auxiliaries are not used

    # Adjust input time series lengths and normalise data
    X = X or {}
    temp_X = {}
    for input_name in model_params['dataSources']:
        if input_name == 'aux':
            continue
        input_params = model_params['inputs'][input_name]
        input_data = X[input_name]
        if input_data is not None:
            input_data = input_data.loc[samples.index]  # Ensure input_data index matches samples
            channels = input_params.get('channels', 0)
            if channels > 0:
                temp_X[input_name] = prepare_ts_data(
                    model_params, input_params, input_data, predict)
                if (augment is True) or (isinstance(augment, list) and input_name in augment):
                    temp_aux = np.concatenate(
                        [temp_aux, temp_X[input_name][:, -1, :]], axis=1)
                elif isinstance(augment, dict) and input_name in augment.keys():
                    offset = augment[input_name] or 1
                    temp_aux = np.concatenate(
                        [temp_aux, temp_X[input_name][:, -offset, :]], axis=1)
            else:
                input_data = np.array(input_data)
                temp_X[input_name] = input_data
            print(f'Prepared {input_name} shape: {temp_X[input_name].shape}')

    if temp_aux is not None:
        temp_X['aux'] = temp_aux
        print(f'Prepared aux shape: {temp_X["aux"].shape}')
        
    if model_params['parentModel'] and model_params['parentResult']:
        temp_X['parent'] = np.zeros((samples.shape[0], 1))
        
    return temp_X


def create_models(model_params, samples=None, X={}, y=None, parent_model=None):
    """Creates a set of LFMC models.
    
    Creates a set of LFMC models. The number of models (or model runs)
    controlled by the modelRuns model parameter.
    
    If only a single run is requested, the model parameters are passed
    unchanged to either ``train_test_model`` (after splitting the data)
    or ``run_kfold_model``.
    
    If multiple runs are requested, the model parameters are modified
    to set a unique output directory for each run and to set the random
    seed (using the seedList). If required, the data is re-split
    between each run. After all runs are completed, aggregate
    evaluation statistics are created. If the same splits were used for
    each run, ensembles of equivalent models from all runs are created
    and evaluated.

    To save memory, the Keras components are removed from the models
    before they are returned. If the models need to be kept, ensure
    the ``saveModels`` model parameter is ``True``.
        
    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters.
    samples : DataFrame, optional
        A data frame containing the samples. If `None` the samples will
        be loaded from the source file. The default is None.
    X : dict, optional
        The predictor data. The keys are the name of each source. Each
        value is a DataFrame of the predictors for that source. If an
        empty dictionary or any sources are missing, these sources will
        be read from the source files. Any sources set to `None` will
        be skipped. The default is {}.
    y : pd.Series, optional
        The labels. A series containing the label for each sample.
        If `None` the labels will be extracted from the samples. The
        default is None.

    Returns
    -------
    LfmcModel or ModelList
        The trained and evaluated model or models. 

    """
    
    def pool_initialiser(gpu_queue):
        gpu = gpu_queue.get()
        train_test_model.gpu = gpu

    def create_worker_pool(model_params):
        num_pool_workers = model_params.get('maxWorkers', 1)
        gpu_list = model_params.get('gpuList', [])
        if gpu_list:
            gpu_queue = Manager().Queue()
            for gpu in (gpu_list * num_pool_workers)[:num_pool_workers]:
                gpu_queue.put(gpu)
            init = pool_initialiser
        else:
            init = None
            gpu_queue = None
        return get_context('spawn').Pool(num_pool_workers, init, (gpu_queue,))

    def setup_runs(model_params):
        restart = model_params.get('restartRun', 0)
        if restart:
            models = ModelList.load_model_set(model_dir, num_runs=restart)
            file_name = f'model_params{restart}.json'
            if len(models) < restart:
                warnings.warn(f'Restart at run {restart} requested, but only {len(models)} ' \
                              'runs found. Missing runs ignored.')
            else:
                print(f'Restarting at run {restart}')
        else:
            restart = 0
            models = ModelList(model_params['modelName'], model_dir)
            file_name = 'model_params.json'
        model_params.save(file_name)
        return models, restart

    def merge_run_results(models, model_params):
        model_dir = model_params['modelDir']
        gen_test_results(model_dir, models)
        can_ensemble = ((not model_params['resplit']
                         or model_params['splitMethod'] == 'byYear')
                       and (model_params['testFolds'] < 2))
        if can_ensemble:
            create_ensembles(model_dir, models)
        if model_params['evaluateEpochs']:
            for epoch in models[0].epoch_results.keys():
                print(f"\nResults summary: {epoch}\n{'-' * (17 + len(epoch))}")
                gen_test_results(model_dir, models, epoch=epoch)
                if can_ensemble:
                    create_ensembles(model_dir, models, epoch=epoch)
    
    def samples_filter(parent_filter, samples, parent_results):
        filter_method = parent_filter[0]
        filter_params = parent_filter[1:]
        filter_str = ', '.join([str(x) for x in filter_params])
        print(f"Samples filtered on parent results using \"{filter_method}({filter_str})\"")
        filter_values = parent_results.drop(columns='y').mean(axis=1)
        filter_ = getattr(filter_values, filter_method)(*filter_params)
        return samples[filter_.reindex(samples.index)]
        
        
    # Add attributes to kfold_split to store the fold indexes between runs
    kfold_split.indexes = None
    kfold_split.folds = None

    model_dir = model_params['modelDir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    samples, X = _read_data(model_params, samples, X)
    if y is None:
        y = prepare_labels(model_params, samples)
    X = prepare_data(model_params, samples, X)
    if model_params['splitMethod'] == 'byYear':
        has_folds = model_params.get('yearFolds', None)
        has_folds = (has_folds if isinstance(has_folds, int) else model_params['splitFolds']) > 1
    else:
        has_folds = model_params['splitMethod'] and (model_params['splitFolds'] > 1)

    with contextlib.closing(create_worker_pool(model_params)) as pool:
        # Build and run several models
        if model_params['modelRuns'] >= 1:
            models, restart = setup_runs(model_params)
            data = None
            for run in range(restart, model_params['modelRuns']):
                parent_results = None if parent_model is None else parent_model[run].all_results
                run_params = _set_run_params(model_params, run)
                if not has_folds: # model_params['splitFolds'] <= 1:
                    if data is None or model_params['resplit']:
                        data = train_test_split(run_params, samples, X, y, parent_results)
                    model = _train_test_subprocess(pool, model_params=run_params, **data)
                else:
                    model = run_kfold_model(pool, run_params, samples, X, y, parent_results)
                models.append(model)
                model_params['plotModel'] = False  # Disable model plotting after the first run
                    
            if not has_folds:  # model_params['splitFolds'] <= 1:
                for num, model in enumerate(models[restart:], restart):
                    m = model.get()
                    models[num] = m
            else:
                merge_run_results(models, model_params)
    
        # Build and run a single model
        elif model_params['modelRuns'] == 0:
            parent_results = None if parent_model is None else parent_model.all_results
            if not has_folds: # model_params['splitFolds'] <= 1:
                data = train_test_split(model_params, samples, X, y, parent_results)
                models = _train_test_subprocess(pool, model_params=model_params, **data).get()
        
            # Build and run a k-fold model
            else:    # model_params['splitFolds'] > 1:
                models = run_kfold_model(pool, model_params, samples, X, y, parent_results)
    
    return models
    
    
def run_experiment(experiment, model_params, samples=None, X={}, y=None):
    """Runs an LFMC experiment
    
    Parameters
    ----------
    experiment : dict
        A dictionary of experiment parameters.
    model_params : ModelParams
        The model parameter dictionary.
    samples : DataFrame, optional
        A data frame containing the samples. If `None` the samples will
        be loaded from the source file. The default is None.
    X : dict, optional
        The predictor data. The keys are the name of each source. Each
        value is a DataFrame of the predictors for that source. If an
        empty dictionary or any sources are missing, these sources will
        be read from the source files. Any sources set to `None` will
        be skipped. The default is {}.
    y : pd.Series, optional
        The labels. A series containing the label for each sample.
        If `None` the labels will be extracted from the samples. The
        default is None.

    Returns
    -------
    models : ModelList
        The results of each test in the experiment. Each element is
        either a Model (for single runs, without k-fold splits) or a
        ModelList (for multi-run tests, or tests using k-fold splits).
    """
    def get_test_range(rerun, num_tests, num_models):
        test_range = []
        badtests = []
        missing_tests = []
        for test_num in (rerun if isinstance(rerun, list) else [rerun]):
            if isinstance(test_num, int) and (test_num >= 0) and (test_num < num_tests):
                test_range.append(test_num)
                if test_num >= num_models:
                    missing_tests.append(test_num)
            else:
                badtests.append(test_num)
        if missing_tests:
            warnings.warn(f'Re-run of tests {rerun} requested, but tests {missing_tests} were' \
                          ' not found. These will be run as new tests.')
        if badtests:
            warnings.warn(f'Re-run of tests {rerun} requested, but tests {badtests} are' \
                          ' invalid and will not be run.')
        return sorted(test_range)

    def setup_experiment(experiment, model_params):
        model_dir = model_params['modelDir']
        print_str = f"Experiment {experiment['name']} - {experiment['description']}"
        print_heading(print_str, line_char='=', line_before=True, blank_after=1)
        
        restart = experiment.get('restart')
        rerun = experiment.get('rerun', [])
        num_tests = len(experiment.setdefault('tests', [{}]))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        elif restart is None and not (rerun or rerun == 0):
            raise FileExistsError(f'{model_dir} exists but restart not requested')
        if restart:
            test_range = range(restart, num_tests)
            models = ModelList.load_experiment(model_dir, num_tests=restart)
            if len(models) < restart:
                warnings.warn(f'Restart at test {restart} requested, but only {len(models)} ' \
                              'tests found. Missing tests ignored.')
            experiment_file = f'experiment{restart}.json'
            model_file = f'model_params{restart}.json'
        elif rerun or rerun == 0:
            models = ModelList.load_experiment(model_dir)
            test_range = get_test_range(rerun, num_tests, len(models))
            experiment_file = f'experiment{test_range[0]}.json'
            model_file = f'model_params{test_range[0]}.json'
        else:
            test_range = range(num_tests)
            models = ModelList(experiment['name'], model_dir)
            experiment_file = 'experiment.json'
            model_file = 'model_params.json'
        experiment.save(experiment_file, model_dir)
        model_params.save(model_file)
        return models, test_range

    models, test_range = setup_experiment(experiment, model_params)
    samples, X = _read_data(model_params, samples, X, inputs_needed=False)
    for test_num in test_range:
        print(f"\n{'-' * 70}\n")
        test_params, reload = _set_test_params(experiment, model_params, test_num)
        parent = test_params.get('parentModel')
        if isinstance(parent, int):
            parent_model = models[parent]
        else:
            parent_model = None
        if (reload == []) or ((samples is None) and (not X) and (y is None)):
            model = create_models(test_params, samples, X, y, parent_model)
        else:
            test_X = {k: v for k, v in X if k not in reload}
            test_samples = None if 'aux' in reload else samples
            model = create_models(test_params, test_samples, test_X, y, parent_model)
        if test_num < len(models):
            models[test_num] = model
        else:
            models.append(model)
        if not experiment.get('resumeAllTests', False):
            model_params['restartRun'] = None  # Turn off any restart after running the first test
    return models