"""Model building and evaluation utilities"""

import glob
import numpy as np
import os
import pandas as pd
import random

from model_list import ModelList
from analysis_utils import calc_statistics
from data_prep_utils import normalise, denormalise
from lfmc_model import import_model_class

   
# =============================================================================
# Process test results functions
# =============================================================================


def merge_kfold_results(model_dir, models, folds, epochs=False):
    """Merges fold predictions
    
    Merges the predictions from each fold to create a predictions file
    for the run. If results from intermediate epochs have been saved
    then the epoch predictions are also merged. Statistics for each
    merged set of predictions are saved.
    
    The merged predictions and stats are saved as ``models`` attributes
    all_results and all_stats. Any results for intermediate epochs are
    saved to the epoch_results and epoch_stats attrbiutes.

    Parameters
    ----------
    model_dir : str
        Name of directory containing the folds, and where the merged
        results will be written.
    models : ModelList
        The kfold models to merge. The model results should be stored
        in the ``all_results`` attribute on each model.
    folds : list
        List of fold names. 
    epochs : bool, optional
        Indicates if results were saved at intermediate epochs. If
        ``True``, the epoch results are also merged. The default is False.

    Returns
    -------
    all_results : dict
        The merged predictions. Columns are ``y`` the labels plus the
        derived model names. Indexed by the sample IDs.
    all_stats : dict
        The prediction statistics. Columns are the statistic type.
        Indexed by the model names.
    """
    def save_results(predictions, output_dir):
#        predictions = pd.concat(list(predictions.values()))
        stats = {}
        for y in predictions.columns.drop('y'):
            stats[y] = calc_statistics(predictions.y, predictions[y])
        all_stats = pd.DataFrame.from_dict(stats, orient='index')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        predictions.to_csv(os.path.join(output_dir, 'predictions.csv'))
        all_stats.to_csv(os.path.join(output_dir, 'predict_stats.csv'))
        return predictions, all_stats
    
    if epochs:
        models.epoch_results = {}
        models.epoch_stats = {}
        for epoch in models[0].epoch_results.keys():
            predictions = [model.epoch_results[epoch] for model in models]
            temp_results = save_results(predictions, os.path.join(model_dir, epoch))
            models.epoch_results[epoch] = temp_results[0]
            models.epoch_stats[epoch] = temp_results[1]
                
    test_all_years = models[0].params['testAllYears']
    year_folds = models[0].params['yearFolds'] or 0
    if test_all_years and (year_folds > 1):
        pred_dict = {}
        for model in models:
            fold = model.params['fold']
            preds =  model.all_results
            year = fold.split('_')[0]
            pred_dict.setdefault(year, [])
            pred_dict[year].append(preds)
        predictions = [pd.concat(vals).drop(columns='y').add_prefix(f'{key}_')
                       for key, vals in pred_dict.items()]
        predictions = pd.concat(predictions, axis=1)
        pred_y = pd.concat([values.y for x in pred_dict.values() for values in x])
        pred_y = pred_y.reset_index().drop_duplicates('ID').set_index('ID')
        predictions.insert(0, 'y', pred_y.y)
    else:
        predictions = pd.concat([model.all_results for model in models])
    all_results, all_stats = save_results(predictions, model_dir)
    models.all_results = all_results
    models.all_stats = all_stats
    return


def gen_test_results(model_dir, models=None, epoch=''):
    """ Generate test statistics.
    
    Generate the summary statistics for the test. This can be done for
    the full model, or for a saved epoch.

    Parameters
    ----------
    model_dir : str
        Directory containing the models (if ``models=None``), also
        where the results are saved.
    models : ModelList, optional
        List of the models created for the test. The function assumes
        each model has all_stats and all_results attributes and adds
        attributes for the test and ensemble results. If ``None``, the
        predictions and stats are loaded from the runs in model_dir.
        The default is ``None``. Use the default to get epoch results.
    epoch : str, optional
        Name of the epoch directory. Required when model evaluation at
        regular epochs is to be done. Test results are generated using
        all runs that saved results at this epoch. If ``''``, the
        results from the full model are used. The default is ``''``.

    Returns
    -------
    None.
    """
    
    if models:
        if epoch == '':
            stack_stats = pd.concat([model.all_stats.stack() for model in models], axis=1)
            output_dir = model_dir
        else:
            stack_stats = pd.concat([model.epoch_stats[epoch].stack() for model in models], axis=1)
            output_dir = os.path.join(model_dir, epoch)
    else:
        output_dir = os.path.join(model_dir, epoch)
        stats_list = glob.glob(os.path.join(model_dir, 'run*', epoch, 'predict_stats.csv'))
        stack_stats = [pd.read_csv(stats_file, index_col=0).stack() for stats_file in stats_list]
        stack_stats = pd.concat(stack_stats, axis=1)
    # Calculate mean and variances of the run prediction statistics
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stack_stats.to_csv(os.path.join(output_dir, 'stats_all.csv'))
    means = stack_stats.mean(axis=1).unstack()
    means.to_csv(os.path.join(output_dir, 'stats_means.csv'), float_format="%.2f")
    variances = stack_stats.var(axis=1).unstack()
    variances.to_csv(os.path.join(output_dir, 'stats_vars.csv'), float_format="%.2f")
    if models and not epoch:
        models.run_stats = stack_stats
        models.means = means
        models.variances = variances


def create_ensembles(model_dir, models='run*', epoch='', ensemble_name='ensemble', precision=2):
    """ Create ensembles for all derived models.
    
    Create and evaluate ensembles of the individual run predictions for
    a test or model set. Ensembles can be created using all runs or for
    a selection of runs. They can be generated from the full models or
    for any saved epoch.

    Parameters
    ----------
    model_dir : str or None
        Directory containing the models if they are to be read from
        disk, also where the results are saved. If None, models must be
        a ModelList or a list of predictions, and ensembles will not be
        saved.
    models : ModelList, list or str, optional
      - If a ``ModelList``: List of the models to ensemble. The function
        assumes each model has all_stats and all_results attributes and
        adds attributes for the test and ensemble results.
      - If a ``str`` or ``list``: A directory or list of directories
        containing the models to ensemble. These are relative to
        ``model_dir`` and may contain wildchars. The directories or
        relevant epoch sub-directories) are assumed to contain a file
        called ``predictions.csv`` that holds the model predictions.
        Any directory without this file is silently ignored. The
        default is 'run*'.
      - If a ``list`` and ``model_dir`` is ``None``: A list of
        prediction dataframes to ensemble.
    epoch : str, optional
        Name of the epoch directory. Required when model evaluation at
        regular epochs is to be done. Ensembles are created using all
        runs that saved results at this epoch. If '', the results from
        the full model are used. The default is ''.
        Note: ``epoch`` is ignored if ``models`` is a ``ModelList``.
    ensemble_name : str, optional
        Name of the ensemble. This is prefixed to the output file names
        (Note: For compatibility, if the default name is used it is
        not prefixed to the ensemble plots). The default is 'ensemble'.

    Returns
    -------
    ensembles : DataFrame
        A dataframe containing the ensemble predictions indexed by the
        sample ids. Columns are the checkpoint (derived) models.
    all_stats : DataFrame
        A dataframe containing the prediction statistics indexed by the
        checkpoint (derived) model. Columns are the statistic type.
    """
    def load_predictions(model_dir, models, epoch):
        models = models or 'run*'
        if isinstance(models, str):
            preds_list = glob.glob(os.path.join(model_dir, models, epoch, 'predictions.csv'))
        else:
            preds_list = []
            for fn in models:
                preds_list.extend(glob.glob(os.path.join(model_dir, fn, epoch, 'predictions.csv')))
        stack_results = [pd.read_csv(preds_file, index_col=0).stack() for preds_file in preds_list]
        stack_results = pd.concat(stack_results, axis=1)
        return stack_results

    def form_ensembles(model_dir, models, epoch):
        if isinstance(models, ModelList):
            if epoch == '':
                stack_results = pd.concat([model.all_results.stack() for model in models], axis=1)
            else:
                stack_results = pd.concat(
                    [model.epoch_results[epoch].stack() for model in models], axis=1)
        elif model_dir is None:   # Assume models is a list of dataframes
            stack_results = pd.concat([model.stack(level=-1) for model in models], axis=1)
#            print(stack_results.shape)
        else:
            stack_results = load_predictions(model_dir, models, epoch);
        return stack_results.mean(axis=1).unstack(), stack_results.std(axis=1).unstack()
    
    def gen_stats(ensembles):
        stats_ = {}
        for y in ensembles.columns.drop('y'):
            stats_[y] = calc_statistics(ensembles.y, ensembles[y], precision=precision)
        return pd.DataFrame.from_dict(stats_, orient='index')

    def save_ensemble(ensembles, ens_stats, model_dir, epoch):
        output_dir = os.path.join(model_dir, epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ensembles.to_csv(os.path.join(output_dir, f'{ensemble_name}_preds.csv'))
        ens_stats.to_csv(os.path.join(output_dir, f'{ensemble_name}_stats.csv'))
        
    ensembles, uncertainty = form_ensembles(model_dir, models, epoch)
    ens_stats = gen_stats(ensembles)
    if model_dir and ensemble_name:
        save_ensemble(ensembles, ens_stats, model_dir, epoch)
    if isinstance(models, ModelList):
        models.all_results = ensembles
        models.all_stats = ens_stats
    else:
        return ensembles, uncertainty, ens_stats


def generate_ensembles(model_predicts, ensemble_runs, ensemble_sizes,
                       precision=2, uncertainty=False, random_seed=99):
    """ Generate sets of ensembles
    
    Can generate sets of ensembles of varying sizes for a test or model
    set, or a set of ensembles of fixed size for all tests in an
    experiment.

    Parameters
    ----------
    model_predicts : list
        Either a list of prediction data frames forming a model set
        Or a list of lists of prediction data frames forming an
        experiment (one list of prediction data frames for each test).
    ensemble_runs : int
        The number of ensembles (for each test) to create.
    ensemble_sizes : int or list
        Either the size of the ensembles or a list of ensemble sizes.
    random_seed : int, optional
        The random number generator seed. The default is 99.

    Returns
    -------
    preds : list
        a list of the predictions made by each ensemble. A list of
        lists if model_predicts is a list of data frames.
    stats : list
        a list of the prediction statistics for each ensemble. A list
        of lists if model_predicts is a list of data frames.

    """    
    def gen_ensemble_set(model_predicts, ensemble_runs, ensemble_size):
        preds = []
        stds = []
        stats = []
        num_models = len(model_predicts)
        if ensemble_size == 1:
            if ensemble_runs < num_models:
                sample_list = random.sample(range(num_models), k=ensemble_runs)
                for sample in sample_list:
                    preds.append(model_predicts[sample])
            else:
                preds = model_predicts
            for pred in preds:
                p_iter = pred.drop('y', axis=1).iteritems()
                s = {p[0]: calc_statistics(pred.y, p[1], precision=precision) for p in p_iter}
                stats.append(pd.DataFrame.from_dict(s, orient='index'))
        else:
            for run in range(ensemble_runs):
                if run % 10 == 9:
                    print(run+1, end='')
                else:
                    print('.', end='')
                sample_list = random.sample(range(num_models), k=ensemble_size)
                pred_list = [model_predicts[s] for s in sample_list]
                ensemble_ = create_ensembles(None, pred_list, ensemble_name='', precision=precision)
                preds.append(ensemble_[0])
                stds.append(ensemble_[1])
                stats.append(ensemble_[2])
        return preds, stds, stats

    random.seed(random_seed)
    all_stats = []
    predict = []
    std_devs = []
    if isinstance(model_predicts[0], pd.DataFrame):
        # Generate ensembles of varying sizes
        if isinstance(ensemble_sizes, int):
            ensemble_sizes = [ensemble_sizes]
        for ensemble_size in ensemble_sizes:
            print(f'Generating ensembles - size {ensemble_size}:', end=' ')
            preds_, stds_, stats_ = gen_ensemble_set(
                model_predicts, ensemble_runs, ensemble_size)
            all_stats.append(stats_)
            predict.append(preds_)
            std_devs.append(stds_)
            print('')
    else:
        # Generate fixed-size ensembles for each test
        if isinstance(ensemble_sizes, list):
            # Limited to one ensemble size if more than one test
            ensemble_sizes = ensemble_sizes[0]
        for test_ in range(len(model_predicts)):
            print(f'Generating ensembles - test {test_}:', end=' ')
            preds_, stds_, stats_ = gen_ensemble_set(
                model_predicts[test_], ensemble_runs, ensemble_sizes)
            all_stats.append(stats_)
            predict.append(preds_)
            std_devs.append(stds_)
            print('')
    if uncertainty:
        return predict, std_devs, all_stats
    else:
        return predict, all_stats


# =============================================================================
# Model training and evaluation functions
# =============================================================================


def train_model(model, train, val):
    """Train the model.
    
    Calls the model train method, then creates the derived models.

    Parameters
    ----------
    model : Lfmc_Model
        The model - must be ready for training/fitting - i.e. built and
        compiled, with any callbacks defined.
    train : dict
        The training dataset.
    val : dict
        The validation dataset.
        
    The train and val dictionaries are all the same format. They have
    two items, ``X`` and ``y`` for the predictors and labels
    respectively. ``X`` is also a dictionary with an item for each source.

    Returns
    -------
    None.
    """
    result = model.train(train['X'], train['y'], val['X'], val['y'])
    model.train_result = result

    # Plot the training history
    if model.params['saveTrain']:
        model.plot_train_hist()
    if model.params['validationSet']:
        model.plot_train_hist(metric='loss')

    # Create the derived models
    derived_models = model.params.get('derivedModels', True)
    if isinstance(derived_models, dict):
        for m_name, m_params_ in derived_models.items():
            m_params = m_params_.copy()
            m_type = m_params.pop('type', '').lower()
            if m_type == 'best':
                model.best_model(m_name, **m_params)
            elif m_type == 'merge':
                model.merge_models(m_name, **m_params)
            elif m_type == 'ensemble':
                model.ensemble_models(m_name, **m_params)
    elif derived_models:
        _ = model.best_model()                      # Extract the best checkpoint model
        model.merge_models('merge10', 10)           # Merge the last 10 checkpoints into a new model
        model.ensemble_models('ensemble10', 10)     # Create an ensembled model of last 10 checkpoints
        _ = model.best_model(n=10, merge=True)      # Merge the best 10 checkpoint models
    
    # Save models, if required
    save_models = model.params['saveModels']
    if save_models:
        if isinstance(save_models, str):
            if save_models.lower() == 'all':
                model.save_to_disk()
            else:
                model.save_to_disk(model_name=save_models)
        elif isinstance(save_models, list):
            model.save_to_disk(model_list=save_models)
        else:
            model.save_to_disk('base')


def evaluate_model(model, data, which=None, test_name=None, train_stats=False, save_preds=True):
    """Evaluates a model.
    
    Evaluates a model using the ``data`` dataset. If the ``which``
    parameter is specified, only that child model is run, otherwise
    all child models are run. The model predictions and prediction
    statistics are saved. They are also stored as the ``all_results``
    and ``all_stats`` model attributes, if no test_name is provided.

    Parameters
    ----------
    model : Lfmc_Model
        The model to be evaluated. It should be trained/fitted.
    data : dict
        The evaluation data. It should have two items, ``X`` and ``y``
        for the predictors and labels respectively. ``X`` is also a
        dictionary with an item for each source.
    which : str, optional
        The child model to run. If ``None``, all child models are run.
        The default is None.
    test_name : str, optional
        The name of the test. Prefixed to the prediction and statistics
        file names. The default is None, meaning the default file names
        are used.
    train_stats : bool, optional
        Indicates if the model training time and counts should be
        stored with the prediction statistics. The default is False.
    save_preds : bool, optional
        Indicates if the predictions should be saved. If ``False`` only
        the prediction statistics are saved. The default is True.

    Returns
    -------
    None.
    """
    model_dir = model.model_dir

    # Evaluate the models
    if which is None:
        which = ['base'] + list(model.derived_models.keys())
    elif isinstance(which, str):
        which = [which]
    results = {}
    for model_name in which:
        result_ = model.evaluate(data['X'], data['y'], model_name, plot=False)
        normal = model.params['targetNormalise']
        if normal:
            result_['predict'] = denormalise(result_['predict'], **normal)
            y = denormalise(data['y'], **normal)
        else:
            y = data['y']
        results[model_name] = result_

    # Create dataframes for predictions and stats
    all_results = pd.DataFrame({'y': y, # data['y'],
                                **{name: result['predict'] for name, result in results.items()}})
    all_stats = pd.DataFrame([r['stats'] for r in results.values()], index=results.keys())
    all_stats['sampleCount'] = data['y'].shape[0]
    all_stats['runTime'] = [r['runTime'] for r in results.values()]
    if train_stats:
        all_stats['trainTime'] = model.train_result['runTime']
        all_stats['buildTime'] = model.build_time
        weights = model.weight_counts() 
        all_stats['trainableWeights'] = weights[0]
        all_stats['nonTrainableWeights'] = weights[1]

    if test_name is None:
        # These are the main test results
        preds_file = 'predictions.csv'
        stats_file = 'predict_stats.csv'
        if model.last_epoch == model.params['epochs']:
            model.all_results = all_results
            model.all_stats = all_stats
        else:
            model.epoch_results[f'epoch{model.last_epoch}'] = all_results
            model.epoch_stats[f'epoch{model.last_epoch}'] = all_stats
    else:
        preds_file = f'{test_name}_predicts.csv'
        stats_file = f'{test_name}_stats.csv'

    # Save results to CSV files
    if save_preds and (test_name or (model.params.get('fold') is None)):
        # Save predictions if required. The test-set predictions for a "fold" model are not
        # saved, as these will be merged with the other fold results and stored in the "run"
        # directory.
        all_results.to_csv(os.path.join(model_dir, preds_file))
    all_stats.to_csv(os.path.join(model_dir, stats_file))
    

def train_and_evaluate(model, train, val, test):
    """Trains and evaluates the model.
    
    Train the model, evaluate it using the test data, then evaluate
    using the training data, so the model can be checked for
    overfitting.
    
    Parameters
    ----------
    model : Lfmc_Model
        The model - must be ready for training/fitting - i.e. built and
        compiled, with any callbacks defined.
    train : dict
        The training dataset.
    val : dict
        The validation dataset.
    test : dict
        The test dataset.
        
    The train, val and test dictionaries are all the same format. They
    have two items, ``X`` and ``y`` for the predictors and labels
    respectively. ``X`` is also a dictionary with an item for each source.

    Returns
    -------
    None.
    """
    train_model(model, train, val)
    if test['y'] is not None:
        evaluate_model(model, test)
    if val['y'] is not None:
        evaluate_model(model, val, test_name='validation',
                       save_preds=model.params['saveValidation'])
    if model.params['saveTrain'] is not False:  # i.e. is None or True
        evaluate_model(model, train, 'base', 'train',
                       train_stats=True, save_preds=model.params['saveTrain'])


def train_test_model(model_params, train, val, test):
    """Trains and tests a model.
    
    Builds, trains and evaluates an LFMC model. After training the
    model, several derived models are created from the fully-trained
    (base) model:
      - best - a model using the checkpoint with the best training/
        validation loss
      - merge10 - a model created by merging the last 10 checkpoints.
        The checkpoints are merged by averaging the corresponding
        weights from each model.
      - ensemble10 - an ensembled model of the last 10 checkpoints.
        This model averages the predictions made by each model in the
        ensemble to make the final prediction.
      - merge_best10 - similar to the merge10 model, but uses the 10
        checkpoints with the lowest training/validation losses.
          
    All 5 models are evaluated by calculating statistics based on the
    test predictions. Both the predictions and statistics are saved.
    
    To facilitate overfitting analysis, predictions and statistics
    using the training data are generated using the merge10 model.
    
    Parameters
    ----------
    model_params : ModelParams
        The model parameters.
    train : dict
        The training dataset.
    val : dict
        The validation dataset.
    test : dict
        The test dataset.
        
    The train, val and test dictionaries are all the same format. They
    have two items, ``X`` and ``y`` for the predictors and labels
    respectively. ``X`` is also a dictionary with an item for each source.

    Returns
    -------
    model : LfmcModel
        The built model and results.
    """
    def deduplicate(train):
        dedup_found = False
        temp_train = {'X': {}, 'y': None}
        for source, data in train['X'].items():
            if data is None:
                temp_train['X'][source] = None
            else:
                if dedup_found:
                    raise ValueError('Too many sources: only one source can be specified with '
                                     'deduplicate option')
                else:
                    dedup_found = True
                    temp_train['X'][source], index_ = np.unique(data, axis=0, return_inverse=True)
        temp_train['y'] = pd.DataFrame(zip(index_, train['y']), columns = ['ind', 'value']
                                       ).groupby(['ind']).mean()['value']  
        return temp_train

    def normalise_data(train_data, val_data, test_data, model_params):
        model_dir = model_params['modelDir']
        save_params = model_params['saveModels']
        for input_name, input_parms in model_params['inputs'].items():
            normal = input_parms.get('normalise', {}) or {}
            train_data['X'][input_name], params = normalise(
                train_data['X'][input_name],
                **normal,
                input_name=input_name,
                model_dir=model_dir,
                save_params=save_params,
                return_params=True)
            normal.update(params)
            if val_data['X'][input_name] is not None:
                val_data['X'][input_name] = normalise(val_data['X'][input_name], **normal)
            if test_data['X'][input_name] is not None:
                test_data['X'][input_name] = normalise(test_data['X'][input_name], **normal)
        normal = model_params.get('targetNormalise', {}) or {}
        train_data['y'], params = normalise(train_data['y'], **normal,
                                            input_name=input_name,
                                            model_dir=model_dir,
                                            save_params=save_params,
                                            return_params=True)
        normal.update(params)
        if val_data['y'] is not None:
            val_data['y'] = normalise(val_data['y'], **normal)
        if test_data['y'] is not None:
            test_data['y'] = normalise(test_data['y'], **normal)

    def augment_data(train_data, val_data, test_data, model_params):
        aug = model_params['auxAugment']
        if not aug:
            return
        for input_ in [train_data, val_data, test_data]:
            for input_name, input_data in input_['X'].items():
                if (input_data is not None) and (len(input_data.shape) == 3):
                    if (aug is True) or (isinstance(aug, list) and input_name in aug):
                        input_['X']['aux'] = np.concatenate(
                            [input_['X']['aux'], input_data[:, -1, :]], axis=1)
                    elif isinstance(aug, dict) and input_name in aug.keys():
                        offset = aug[input_name] or 1
                        input_['X']['aux'] = np.concatenate(
                            [input_['X']['aux'], input_data[:, -offset, :]], axis=1)
            
    gpu = getattr(train_test_model, 'gpu', None)
    if isinstance(gpu, (int, list)):
        model_params['gpuDevice'] = gpu

    if model_params.get('deduplicate'):
        train = deduplicate(train)
    model_dir = model_params['modelDir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    normalise_data(train, val, test, model_params)
    augment_data(train, val, test, model_params)
    model = import_model_class(model_params['modelClass'])(model_params, inputs=train['X'])
    
    if model_params['plotModel']:
        outdir = model_dir.rstrip('/\\')
        if os.path.basename(outdir).startswith('fold'):
            outdir = os.path.dirname(outdir)
        if os.path.basename(outdir).startswith('run'):
            outdir = os.path.dirname(outdir)
        model.plot(dir_name=outdir)
    
    try:
        if model_params.get('evaluateEpochs'):
            model.epoch_results = {}
            model.epoch_stats = {}
            last_epoch = model_params['epochs']
            epoch_step = model_params['evaluateEpochs']
            for epoch_stop in range(epoch_step, last_epoch, epoch_step):
                model.last_epoch = epoch_stop
                model.model_dir = os.path.join(model_dir, f'epoch{model.last_epoch}', '')
                if not os.path.exists(model.model_dir):
                    os.makedirs(model.model_dir)
                train_and_evaluate(model, train, val, test)
            model.model_dir = model_dir
        model.last_epoch = model_params['epochs']
        train_and_evaluate(model, train, val, test)
    finally:
        model.clear_model()
    return model
