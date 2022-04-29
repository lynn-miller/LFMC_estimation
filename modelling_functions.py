"""Functions to build and evaluate LFMC models"""

import contextlib
import glob
import numpy as np
import os
import warnings

from multiprocess import get_context
from sklearn.preprocessing import OneHotEncoder
from time import sleep

from model_list import ModelList
from model_parameters import ModelParams
from model_utils import partition_by_year, normalise
from model_utils import random_split, split_by_site, kfold_indexes, split_data, train_val_split
from model_utils import train_test_model, gen_test_results, create_ensembles, merge_kfold_results


def train_test_split(model_params, samples, X, y):
    """Splits data into test and training sets

    Splits data into test, training and optionally validation sets. The
    model parameters determine how to split the data.
    
    Note: this function makes a single train/test split. kfold_split
    is used if multiple train/test splits are required.
    
    validationSet:
        Indicates if a validation set should be created.
    
    splitMethod:
      - random: samples are split randomly
      - bySite: sites are split into test and training sites, then all
        samples from the site are allocated to the respective set
      - byYear: samples are allocated to the test and training sets
        based on the sampling year
    
    splitSizes:
        The proportion or number of samples to allocate to the test and
        validation sets. Used for random and bySite splits.
    
    siteColumn:
        The column in the samples containing the site. Used for bySite
        splits.
    
    yearColumn:
        The column in the samples containing the sampling year. Used
        for byYear splits.
    
    splitStratify:
        Indicates the column to use for stratified splitting, if this
        is needed. Stratified splitting groups the sites by value and
        ensures each split contains a roughly equal proportion of each
        group. If None, stratified splittig is not used. Used for
        bySite splits.
    
    splitYear:
      - If splitMethod is "byYear" and splitFolds is 0: Samples for
        this year or later go into the test set. Samples earlier than
        this year go into the training set.
      - If splitMethod is "byYear" and splitFolds is 1: Samples for
        this year go into the test set. Samples earlier than this year
        go into the training set. Samples for later than this year are
        not used.
      - If splitMethod is "random" or "bySite": Any samples in the test
        set for earlier than this year are removed. Any samples in the
        training or validation sets for this year or later are removed.
                
    randomSeed:
        The seed for the random number generator. Used for random and
        bySite splits.
    
    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters, used as described above.
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
    data : dict
        The split data with an item for each dataset. Keys are ``test``,
        ``val(idation)``, or ``train(ing)``. Values are dictionaries
        with two entries - ``X`` and ``y``. The ``X`` values are again
        dictionaries, with entries for each source. The lowest level
        values are numpy arrays containg the relevant data. ``val``
        entries are included even when no validation set is required.
        In this case, all lowest level values are ``None``.
    """
    if model_params['splitMethod'] == 'random':
        train_index, val_index, test_index = random_split(
                num_samples=samples.shape[0],
                split_sizes=model_params['splitSizes'],
                val_data=model_params['validationSet'],
                random_seed=model_params['randomSeed'])
        if model_params['splitYear']:
            train_index, val_index, test_index = partition_by_year(
                np.array(samples[model_params['yearColumn']]),
                model_params['splitYear'],
                train_index, val_index, test_index)
    elif model_params['splitMethod'] == 'bySite':
        train_index, val_index, test_index = split_by_site(
                data=samples,
                split_sizes=model_params['splitSizes'],
                split_col=model_params['siteColumn'],
                stratify=model_params['splitStratify'],
                val_data=model_params['validationSet'],
                random_seed=model_params['randomSeed'])
        if model_params['splitYear']:
            train_index, val_index, test_index = partition_by_year(
                np.array(samples[model_params['yearColumn']]),
                model_params['splitYear'],
                train_index, val_index, test_index)
    elif model_params['splitMethod'] == 'byYear':
        sample_years = np.array(samples[model_params['yearColumn']])
        if model_params['splitFolds'] == 1:
            test_index = sample_years == model_params['splitYear']
        else:  # splitFolds will be 0 or None
            test_index = sample_years >= model_params['splitYear']
        if model_params['validationSet']:
            train_val_index = sample_years < model_params['splitYear']
            train_index, val_index = train_val_split(
                    num_samples=samples.shape[0], #train_val_index.sum(),
                    val_size=model_params['splitSizes'][1],
                    train_val_idx=train_val_index,
                    random_seed=model_params['randomSeed'])
        else:
            train_index = sample_years < model_params['splitYear']
            val_index = None
    elif not model_params['splitMethod']:    # No test set needed
        test_index = None
        train_val_index = np.array([True] * samples.shape[0])
        if model_params['validationSet']:
            train_index, val_index = train_val_split(
                    num_samples=samples.shape[0],
                    val_size=model_params['splitSizes'][1],
                    train_val_idx=train_val_index,
                    random_seed=model_params['randomSeed'])
        else:
            train_index = train_val_index
            val_index = None
    else:
        raise ValueError(f"Invalid train/test split method: {model_params['splitMethod']}")

    y_train, y_test, y_val = split_data(y, train_index, test_index, val_index)
    data = {'train': {'X': {}, 'y': y_train},
            'val':   {'X': {}, 'y': y_val},
            'test':  {'X': {}, 'y': y_test}}
    for source, xdata in X.items():
        train, test, val = split_data(xdata, train_index, test_index, val_index)
        data['train']['X'][source] = train
        data['val']['X'][source] = val
        data['test']['X'][source] = test
    return data


def kfold_split(model_params, samples, X, y):
    """Splits data into folds

    A generator that splits data into folds containing test, training
    and optionally validation sets. The samples are split into ``K``
    roughly equal parts (where ``K`` is the number of folds), then
    folds are formed by using each part in turn as the test set and the
    remaining parts as the training set. The validation set (if used)
    is randomly selected from the training set. The generator yields
    each fold in turn.

    The following model parameters determine how to split the data. 

    validationSet:
        Indicates if a validation set should be created.
    
    splitMethod:
      - random: samples are split randomly
      - bySite: sites are split into test and training sites, then all
        samples from the site are allocated to the respective set
      - NOTE: byYear cannot be used with k-fold splits
    
    splitFolds:
        The number of folds (``K``) required.
        
    splitSizes:
        The proportion or number of samples to allocate to validation
        sets. This must contain two values (consistent with standard
        splitting) but the first value is ignored.
    
    siteColumn:
        The column in the samples containing the site. Used for bySite
        splits.
    
    splitStratify:
        Indicates the column to use for stratified splitting, if this
        is needed. Stratified splitting groups the sites by value and
        ensures each split contains a roughly equal proportion of each
        group. If None, stratified splittig is not used. Used for
        bySite splits.
    
    randomSeed:
        The seed for the random number generator.
    
    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters, used as described above.
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

    Yields
    ------
    data : dict
        The split data with an item for each dataset. Keys are ``test``,
        ``val(idation)``, or ``train(ing)``. Values are dictionaries
        with two entries - ``X`` and ``y``. The ``X`` values are again
        dictionaries, with entries for each source. The lowest level
        values are numpy arrays containg the relevant data. ``val``
        entries are included even when no validation set is required.
        In this case, all lowest level values are ``None``.
    """
    if not kfold_split.folds or model_params['resplit']:
        val_size = model_params['splitSizes'][1] if model_params['validationSet'] else 0
        if model_params['splitMethod'] in ['random', 'bySite']:
            folds = range(model_params['splitFolds'])
            if model_params['splitMethod'] == 'random':
                train_index, val_index, test_index = kfold_indexes(
                        data=samples.reset_index(),
                        n_folds=model_params['splitFolds'],
                        split_col=samples.index.name or 'index',
                        stratify=False,
                        val_size=val_size,
                        random_seed=model_params['randomSeed'])
            else:   # model_params['splitMethod'] == 'bySite'
                train_index, val_index, test_index = kfold_indexes(
                        data=samples,
                        n_folds=model_params['splitFolds'],
                        split_col=model_params['siteColumn'],
                        stratify=model_params['splitStratify'],
                        val_size=val_size,
                        random_seed=model_params['randomSeed'])
            if model_params['splitYear']:
                for n in folds:
                    train_index[n], val_index[n], test_index[n] = partition_by_year(
                        np.array(samples[model_params['yearColumn']]),
                        model_params['splitYear'],
                        train_index[n], val_index[n], test_index[n])
        elif model_params['splitMethod'] == 'byYear':
            train_index = {}
            val_index = {}
            test_index = {}
            sample_years = np.array(samples[model_params['yearColumn']])
            first_year = model_params['splitYear']
            last_year = first_year + model_params['splitFolds']
            folds = list(reversed(range(first_year, last_year)))
            for split_year in range(first_year, last_year):
                test_index[split_year] = sample_years == split_year
                if model_params['validationSet']:
                    train_val_index = sample_years < split_year
                    ti, vi = train_val_split(
                            num_samples=samples.shape[0], #train_val_index.sum(),
                            val_size=val_size,
                            train_val_index=train_val_index,
                            random_seed=model_params['randomSeed'])
                    train_index[split_year] = ti
                    val_index[split_year] = vi
                else:
                    train_index[split_year] = sample_years < split_year
                    val_index[split_year] = None
        elif not model_params['splitMethod']:
            raise ValueError(
                'Invalid train/test split: "splitFolds" > 1 specified with no "splitMethod"')
        else:
            raise ValueError(f"Invalid train/test split method: {model_params['splitMethod']}")
        kfold_split.folds = folds
        kfold_split.indexes = {'train': train_index, 'val': val_index, 'test': test_index}
    else:
        folds = kfold_split.folds
        train_index = kfold_split.indexes['train']
        val_index = kfold_split.indexes['val']
        test_index = kfold_split.indexes['test']

    for fold in folds:
        y_train, y_test, y_val = split_data(
            y, train_index[fold], test_index[fold], val_index[fold])
        data = {'train': {'X': {}, 'y': y_train},
                'val':   {'X': {}, 'y': y_val},
                'test':  {'X': {}, 'y': y_test}}
        for source, xdata in X.items():
            train, test, val = split_data(
                xdata, train_index[fold], test_index[fold], val_index[fold])
            data['train']['X'][source] = train
            data['val']['X'][source] = val
            data['test']['X'][source] = test
        yield (fold, data)

# Add attributes to kfold_split to store the fold indexes between runs
kfold_split.indexes = None
kfold_split.folds = None


def train_test_subprocess(pool, **kwargs):
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


def run_kfold_model(pool, model_params, samples, X, y):
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
    models = ModelList(model_name, model_dir)

    folds = []
    for fold, data in kfold_split(model_params, samples, X, y):
        folds.append(f'fold{fold}')
        fold_params = ModelParams(model_params.copy())
        fold_params['modelName'] = f"{model_name}_fold{fold}"
        fold_params['modelDir'] = os.path.join(model_dir, f'fold{fold}')
        if not os.path.exists(fold_params['modelDir']):
            os.makedirs(fold_params['modelDir'])
        model = train_test_subprocess(pool, model_params=fold_params, **data)
        models.append(model)

    for num, model in enumerate(models):
        models[num] = model.get()
         
    all_results, all_stats = merge_kfold_results(
        model_dir, models, folds=folds, epochs=model_params['evaluateEpochs'])
    models.all_results = all_results
    models.all_stats = all_stats
        
    return models


def prepare_data(model_params, samples, X, fit_data=None):
    """Prepares data for LFMC test.
    
    Prepares the data for an LFMC model test.
    - Extract the auxiliary columns from the sample data
    - One-hot encodes any categorical auxiliary columns
    - Normalises the timeseries data using ``model_params`` settings
    - Augments the auxiliary data with the last item in each time
    series if required

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
    dict
        A dictionary of the prepared data. Dictionary keys are the input
        names and values are numpy array of each input.

    """
    temp_X = {}
    if 'aux' in model_params['dataSources']:
        # Process auxiliary data here so auxiliary inputs can change between tests
        augment = model_params['auxAugment']
        if isinstance(model_params['auxColumns'], int):
            aux_start = len(samples.columns) - model_params['auxColumns']
            x_aux = np.array(samples.iloc[:, aux_start:])
            print('Auxiliary columns:', list(samples.iloc[:, aux_start:].columns))
        else:
            x_aux = np.array(samples[model_params['auxColumns']])
            print('Auxiliary columns:', list(samples[model_params['auxColumns']].columns))
        onehot_cols = model_params['auxOneHotCols']
        if onehot_cols:
            onehot_enc = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype='int')
            if fit_data is None:
                fit_data = samples[onehot_cols]
            onehot_enc.fit(fit_data.dropna().to_numpy())
            onehot_data = onehot_enc.transform(samples[onehot_cols].fillna('').to_numpy())
            x_aux = np.concatenate([x_aux, onehot_data], axis=1)
        temp_aux = x_aux
    else:
        temp_aux = None   # No auxiliary data
        augment = False   # Ignore the auxAugment setting if auxiliaries are not used

    # Adjust input time series lengths and normalise data
    for input_name, input_data in X.items():
        if input_name in model_params['dataSources']:
            if input_data.ndim == 3:
                normal = model_params[input_name + 'Normalise']
                temp_X[input_name] = normalise(input_data, **normal)
                print(f'{input_name} shape: {temp_X[input_name].shape}')
                if model_params['diagnostics']:
                    for x in range(0, temp_X[input_name].shape[1], 24):
                        print(temp_X[input_name][0, x:x+24, 0])
                if (augment is True) or (isinstance(augment, list) and input_name in augment):
                    temp_aux = np.concatenate([temp_aux, temp_X[input_name][:, -1, :]], axis=1)
                elif isinstance(augment, dict) and input_name in augment.keys():
                    offset = augment[input_name] or 1
                    temp_aux = np.concatenate(
                        [temp_aux, temp_X[input_name][:, -offset, :]], axis=1)
            elif input_name != 'aux':
                temp_X[input_name] = input_data
                print(f'{input_name} shape: {temp_X[input_name].shape}')

    if temp_aux is not None:
        temp_X['aux'] = temp_aux
        print(f'aux shape: {temp_X["aux"].shape}')
    return temp_X


def create_models(model_params, samples, X, y):
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
    LfmcModel or ModelList
        The trained and evaluated model or models. 

    """
    
    # Add attributes to kfold_split to store the fold indexes between runs
    kfold_split.indexes = None
    kfold_split.folds = None

    model_dir = os.path.join(model_params['modelDir'], '')

    X = prepare_data(model_params, samples, X)

    num_pool_workers = model_params.get('maxWorkers', 1) 
    with contextlib.closing(get_context('spawn').Pool(num_pool_workers)) as pool:
        # Build and run several models
        if model_params['modelRuns'] > 1:
            model_name = model_params['modelName']
            model_seeds = model_params['seedList']
            if model_seeds is None or len(model_seeds) == 0:
                model_seeds = [model_params['modelSeed']]
            data = None
            restart = model_params.get('restartRun', 0)
            if restart:
                models = ModelList.load_model_set(model_dir, num_runs=restart)
                if len(models) < restart:
                    warnings.warn(f'Restart at run {restart} requested, but only {len(models)} ' \
                                  'runs found. Missing runs ignored.')
                else:
                    print(f'Restarting at run {restart}')
            else:
                restart = 0
                models = ModelList(model_name, model_dir)
            for run in range(restart, model_params['modelRuns']):
                run_params = ModelParams(model_params.copy())
                run_params['modelName'] = f"{model_name}_run{run}"
                run_params['modelDir'] = os.path.join(model_dir, f"run{run}")
                run_params['modelSeed'] = model_seeds[run] if run < len(model_seeds) else None
                if model_params['resplit']:
                    if model_params['seedList']:
                        run_params['randomSeed'] = run_params['modelSeed']
                    elif run > 0:
                        run_params['randomSeed'] = None
                run_params.save('model_params.json')
                if model_params['splitFolds'] <= 1:
                    if data is None or model_params['resplit']:
                        data = train_test_split(run_params, samples, X, y)
                    model = train_test_subprocess(pool, model_params=run_params, **data)
                    models.append(model)
                else:
                    models.append(run_kfold_model(pool, run_params, samples, X, y))
                model_params['plotModel'] = False  # Disable model plotting after the first run
                    
            if model_params['splitFolds'] <= 1:
                for num, model in enumerate(models[restart:], restart):
                    m = model.get()
                    models[num] = m
    
            if model_params['splitMethod']:
                gen_test_results(model_dir, models)
                can_ensemble = (not model_params['resplit']
                                or model_params['splitMethod'] == 'byYear'
                                or model_params['splitFolds'] > 1)
                if can_ensemble:
                    create_ensembles(model_dir, models)
                if model_params['evaluateEpochs']:
                    epoch_dirs = glob.glob(os.path.join(model_dir, 'run*', 'epoch*'))
                    epoch_dirs = sorted({os.path.basename(fn) for fn in epoch_dirs},
                                        key=lambda x: int(x.strip('epoch')))
                    for epoch in epoch_dirs:
                        print(f"\nResults summary: {epoch}\n{'-' * (17 + len(epoch))}")
                        gen_test_results(model_dir, epoch=epoch)
                        if can_ensemble:
                            create_ensembles(model_dir, epoch=epoch)
    
        # Build and run a single model
        elif model_params['splitFolds'] <= 1:
            data = train_test_split(model_params, samples, X, y)
            models = train_test_subprocess(pool, model_params=model_params, **data).get()
    
        # Build and run a k-fold model
        else:    # model_params['splitFolds'] > 1:
            models = run_kfold_model(pool, model_params, samples, X, y)
    
    return models


def set_test_params(experiment, model_params, test_num):
    test = experiment['tests'][test_num]
    try:
        test_name = experiment['testNames'][test_num]
    except:
        test_name = f"{test_num}"
    print(f'Test {test_name} - {test}\n')
    test_params = ModelParams(model_params.copy())
    test_params['testName'] = test_name
    test_params['modelName'] = '_'.join([test_params['modelName'], f"test{test_num}"])
    test_params['modelDir'] = os.path.join(test_params['modelDir'], f"test{test_num}")
    test_params.update(test)
    # Update model layers
    for layer in experiment['layerTypes'] & test.keys():
        layer_parms = experiment.get(layer, {}).copy()
        layer_parms.update(test[layer])
        test_params.set_layers(
            **{layer.replace('Conv', '') + '_layers': layer_parms['numLayers']})
        for key, value_list in layer_parms.items():
            if key != 'numLayers':
                for layer_num, value in enumerate(value_list):
                    test_params[layer][layer_num][key] = value
    return test_params
    
    
def run_experiment(experiment, model_params, samples, X, y):
    """Runs an LFMC experiment
    

    Parameters
    ----------
    experiment : dict
        A dictionary of experiment parameters.
    model_params : ModelParams
        The model parameter dictionary.
    layer_types : list-like
        A list of the layer_types/blocks used.
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
    ex_models : ModelList
        The results of each test in the experiment. Each element is
        either a Model (for single runs, without k-fold splits) or a
        ModelList (for multi-run tests, or tests using k-fold splits).
    """
    model_name = model_params['modelName']
    model_dir = model_params['modelDir']
    print_str = f"Experiment {experiment['name']} - {experiment['description']}"
    print(f"{'=' * len(print_str)}\n{print_str}\n{'=' * len(print_str)}\n")
    restart = experiment.get('restart', 0) or 0
    if restart == 0:
        ex_models = ModelList(model_name, model_dir)
        file_name = 'model_params.json'
    else:
        ex_models = ModelList.load_experiment(model_dir, num_tests=restart)
        if len(ex_models) < restart:
            warnings.warn(f'Restart at test {restart} requested, but only {len(ex_models)} tests' \
                          ' found. Missing tests ignored.')
        file_name = f'model_params{restart}.json'
    model_params.save(file_name)

    for test_num in range(restart, len(experiment['tests'])):
        if test_num != 0:
            print(f"\n{'-' * 70}\n")
        test_params = set_test_params(experiment, model_params, test_num)
        test_params.save('model_params.json')
        ex_models.append(create_models(test_params, samples, X, y))
        if not experiment.get('resumeAllTests', False):
            model_params['restartRun'] = None  # Turn off any restart after running the first test
    return ex_models