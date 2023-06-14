"""Model results processing utilities"""

import glob
import os
import pandas as pd
import random

from model_list import ModelList
from analysis_utils import calc_statistics

   
# =============================================================================
# Process test results functions
# =============================================================================


def save_source_stats(model, results, which, test_name, model_dir, epoch_key=None, target_column='y'):
    sources = results.index.get_level_values('Source').unique()
    source_stats = []
    source_idx = []
    m = model
    while isinstance(m, ModelList):
        m = m[0]
    classify = m.params['classify']
    for s in sources:
        src_results = results.loc[s]
        for model_name in which:
            temp = src_results[[target_column, model_name]].dropna()
            source_stats.append(
                calc_statistics(temp[target_column], temp[model_name], classify=classify))
            source_idx.append((s, model_name))
    source_idx = pd.MultiIndex.from_tuples(source_idx, names=['Source', 'Model'])
    source_stats = pd.DataFrame(source_stats, index=source_idx)
    # Save the results as model attributes
    if epoch_key:
        if hasattr(model, f'epoch_{test_name}_source_stats'):
            getattr(model, f'epoch_{test_name}_source_stats')[epoch_key] = source_stats
        else:
            setattr(model, f'epoch_{test_name}_source_stats', {epoch_key: source_stats})
    else:
        setattr(model, f'{test_name}_source_stats', source_stats)
    if model_dir:
        stats_file = f'{test_name}_source_stats.csv'
#        print(f'Saving file {os.path.join(model_dir, stats_file)}')
        source_stats.to_csv(os.path.join(model_dir, stats_file))


def merge_kfold_results(model_dir, models, epochs=False):
    """Merges fold predictions
    
    Merges the predictions from each fold to create a predictions file
    for the run. If results from intermediate epochs have been saved
    then the epoch predictions are also merged. Statistics for each
    merged set of predictions are saved.
    
    The merged predictions and stats are saved as ``models`` attributes
    *_predicts and *_stats. Any results for intermediate epochs are
    saved to the epoch_*_predicts and epoch_*_stats attrbiutes.

    Parameters
    ----------
    model_dir : str
        Name of directory containing the folds, and where the merged
        results will be written.
    models : ModelList
        The kfold models to merge. The model results should be stored
        in the ``*_predicts`` and ``*_stats`` attributes on each model.
    epochs : bool, optional
        Indicates if results were saved at intermediate epochs. If
        ``True``, the epoch results are also merged. The default is False.
    """
    def get_stats(predictions, classify=False):
        stats = {}
        for y in predictions.columns.drop('y'):
            stats[y] = calc_statistics(predictions.y, predictions[y], classify=classify)
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        return stats_df
    
    def get_test_predictions(models, epoch=None):
        test_all_years = models[0].params['testAllYears']
        year_folds = models[0].params['yearFolds'] or 0
        if test_all_years and (year_folds > 1):
            pred_dict = {}
            for model in models:
                preds = model.test_predicts if epoch is None else model.epoch_test_predicts[epoch]
                year = model.params['fold'].split('_')[0]
                pred_dict.setdefault(year, [])
                pred_dict[year].append(preds)
            predictions = [pd.concat(vals).drop(columns='y').add_prefix(f'{key}_')
                           for key, vals in pred_dict.items()]
            predictions = pd.concat(predictions, axis=1)
            pred_y = pd.concat([values.y for x in pred_dict.values() for values in x])
            pred_y = pred_y.reset_index().drop_duplicates('ID').set_index('ID')
            predictions.insert(0, 'y', pred_y.y)
        else:
            if epoch is None:
                predictions = pd.concat([model.test_predicts for model in models])
            else:
                predictions = pd.concat([model.epoch_test_predicts[epoch] for model in models])
        return predictions

    def save_fold_results(models, result_type, epoch='', save_preds=True):
        output_dir = os.path.join(model_dir, epoch)
        stats_attr = f'{result_type}_stats'
        preds_attr = f'{result_type}_predicts'
        stats_ = []
        preds_ = []
        targets = []
        if epoch == '':
            try:
                stats_ = [getattr(model, stats_attr) for model in models]
                preds_ = [getattr(model, preds_attr).drop(columns='y') for model in models]
                targets = [getattr(model, preds_attr).y for model in models]
            except:
                pass
        else:
            for model in models:
                if epoch in getattr(model, f'epoch_{stats_attr}').keys():
                    stats_.append(getattr(model, f'epoch_{stats_attr}')[epoch])
                    preds_.append(getattr(model, f'epoch_{preds_attr}')[epoch].drop(columns='y'))
                    targets.append(getattr(model, f'epoch_{preds_attr}')[epoch].y)
        if len(stats_) > 0:
            stats_ = pd.concat(stats_, keys=folds)
            stats_.to_csv(os.path.join(output_dir, f'fold_{result_type}_stats.csv'))
            targets = pd.concat(targets)
            target_index = targets.index.names
            targets = targets.reset_index().drop_duplicates(target_index).set_index(target_index)
            preds_ = pd.concat([targets] + preds_, axis=1, keys=['target'] + folds)
            if save_preds:
                preds_.to_csv(os.path.join(output_dir, f'fold_{result_type}_predicts.csv'))
            
        # If there are multiple data sources, save separate stats for each source
        if isinstance(preds_, pd.DataFrame) and preds_.index.nlevels > 1:
            save_source_stats(models, preds_, list(preds_.columns.drop(('target', 'y'))),
                              f'fold_{result_type}', output_dir, epoch, target_column=('target', 'y'))

        
##########
    
    for num, model in enumerate(models):
        models[num] = model.get()
    folds = [model.params['fold'] for model in models]
    params = models[0].params
    classify = params['classify']
    save_runs = params['saveRunResults']
    if epochs:
        models.epoch_test_predicts = {}
        models.epoch_test_stats = {}
        models.epoch_train_predicts = {}
        models.epoch_train_stats = {}
        models.epoch_val_predicts = {}
        models.epoch_val_stats = {}
        for epoch in models[0].epoch_test_predicts.keys():
            if save_runs:
                output_dir = os.path.join(model_dir, epoch)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            else:
                output_dir = None
            predictions = get_test_predictions(models, epoch)
            models.epoch_test_predicts[epoch] = predictions
            models.epoch_test_stats[epoch] = get_stats(predictions, classify)
            if save_runs:
                epoch_dir = os.path.join(model_dir, epoch)
                predictions.to_csv(os.path.join(epoch_dir, 'test_predicts.csv'))
                models.epoch_test_stats[epoch].to_csv(os.path.join(epoch_dir, 'test_stats.csv'))
                
            # If there are multiple data sources, save separate stats for each source
            if predictions.index.nlevels > 1:
                save_source_stats(models, predictions, list(predictions.columns.drop('y')),
                                  'test', output_dir, epoch)
                
            if save_runs:
                save_fold_results(models, 'test', epoch, save_preds=False)
                save_fold_results(models, 'train', epoch, save_preds=params['saveTrain'])
                save_fold_results(models, 'val', epoch, save_preds=params['saveValidation'])
    predictions = get_test_predictions(models)
    models.test_predicts = predictions
    models.test_stats = get_stats(predictions, classify)
    if save_runs:
        predictions.to_csv(os.path.join(model_dir, 'test_predicts.csv'))
        models.test_stats.to_csv(os.path.join(model_dir, 'test_stats.csv'))
                
    # If there are multiple data sources, save separate stats for each source
    if predictions.index.nlevels > 1:
        save_source_stats(models, predictions, list(predictions.columns.drop('y')),
                          'test', model_dir, None)
                
    if save_runs:
        save_fold_results(models, 'test', save_preds=False)
        save_fold_results(models, 'train', save_preds=params['saveTrain'])
        save_fold_results(models, 'val', save_preds=params['saveValidation'])
    return


def gen_test_results(model_dir, models=None, epoch='', stats_files='test_stats.csv',
                     predicts_files='test_predicts.csv'):
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
        each model has test_stats and test_predicts attributes and adds
        attributes for the test and ensemble results. If ``None``, the
        stats are loaded from the runs in model_dir. The default is
        ``None``.
    epoch : str, optional
        Name of the epoch directory. Required when model evaluation at
        regular epochs is to be done. Test results are generated using
        all runs that saved results at this epoch. If ``''``, the
        results from the full model are used. The default is ``''``.
    stats_files : str, optional
        The filename of the stats files for each run. The same name is
        used for each run. The default is 'test_stats.csv'.

    Returns
    -------
    None.
    """
    
    if models:
        if epoch == '':
            stats_list = [model.test_stats.stack() for model in models]
            predicts_list = [model.test_predicts for model in models]
            output_dir = model_dir
        else:
            stats_list = [model.epoch_test_stats[epoch].stack() for model in models]
            predicts_list = [model.epoch_test_predicts[epoch] for model in models]
            output_dir = os.path.join(model_dir, epoch)
        run_list = [i for i in range(len(models))]
    else:
        output_dir = os.path.join(model_dir, epoch)
        stats_list = glob.glob(os.path.join(model_dir, 'run*', epoch, stats_files))
        stats_list = [pd.read_csv(sf, index_col=0).stack() for sf in stats_list]
        predicts_list = glob.glob(os.path.join(model_dir, 'run*', epoch, predicts_files))
        predicts_list = [pd.read_csv(pf, index_col=0).stack() for pf in predicts_list]

    stack_stats = pd.concat(stats_list, axis=1)
    stack_predicts = pd.concat([predicts_list[0].y] + [p.drop(columns='y') for p in predicts_list],
                               axis=1, keys = ['target'] + run_list)
    # Calculate mean and variances of the run prediction statistics
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stack_stats.to_csv(os.path.join(output_dir, 'stats_all.csv'))
    stack_predicts.to_csv(os.path.join(output_dir, 'predicts_all.csv'))
    means = stack_stats.mean(axis=1).unstack()
    means.to_csv(os.path.join(output_dir, 'stats_means.csv'), float_format="%.2f")
    variances = stack_stats.var(axis=1).unstack()
    variances.to_csv(os.path.join(output_dir, 'stats_vars.csv'), float_format="%.2f")
    if models and not epoch:
        models.run_stats = stack_stats
        models.means = means
        models.variances = variances


# =============================================================================
# Ensemble functions
# =============================================================================


def create_ensembles(model_dir, models='run*', epoch='', classify=False,
                     ensemble_name='ensemble', precision=2):
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
        assumes each model has test_stats and test_predicts attributes
        and adds attributes for the test and ensemble results.
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
    classify: bool, optional
      - True: The labels and predictions are classes
      - False: The labels and predictions are numbers. The default is
        False.
    ensemble_name : str, optional
        Name of the ensemble. This is prefixed to the output file names
        (Note: For compatibility, if the default name is used it is
        not prefixed to the ensemble plots). The default is 'ensemble'.
    precision : int, optional
        The precision of the ensemble statistics. The default is 2.
        

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
        pred_file = os.path.join(epoch, 'test_predicts.csv')
        if isinstance(models, str):
            preds_list = glob.glob(os.path.join(model_dir, models, pred_file))
        else:
            preds_list = []
            for dn in models:
                preds_list.extend(glob.glob(os.path.join(model_dir, dn, pred_file)))
        stack_results = [pd.read_csv(preds_file, index_col=0).stack() for preds_file in preds_list]
        stack_results = pd.concat(stack_results, axis=1)
        return stack_results

    def form_ensembles(model_dir, models, epoch):
        if isinstance(models, ModelList):
            if epoch == '':
                preds_list = [model.test_predicts.stack() for model in models]
            else:
                preds_list = [model.epoch_test_predicts[epoch].stack() for model in models]
            stack_results = pd.concat(preds_list, axis=1)
        elif model_dir is None:   # Assume models is a list of dataframes
            stack_results = pd.concat([model.stack(level=-1) for model in models], axis=1)
        else:
            stack_results = load_predictions(model_dir, models, epoch)
        if classify is True:
            classes = stack_results.astype(float).round().mean(axis=1)
            uncertainty = stack_results.sub(classes, axis=0).pow(2).mean(axis=1).pow(0.5)
            return classes.unstack(), uncertainty.unstack()
        elif classify:  # Classify is the number of classes
            classes = stack_results.mode(axis=1)[0].astype(int)
            uncertainty = (stack_results.T != classes).pow(2).mean().pow(0.5)
            return classes.unstack(), uncertainty.unstack()
        else:
            return stack_results.mean(axis=1).unstack(), stack_results.std(axis=1).unstack()
    
    def gen_stats(ensembles):
        stats_ = {}
        for y in ensembles.columns.drop('y'):
            stats_[y] = calc_statistics(ensembles.y, ensembles[y], classify=classify,
                                        precision=precision)
        return pd.DataFrame.from_dict(stats_, orient='index')

    def save_ensemble(ensembles, ens_stats, model_dir, epoch):
        output_dir = os.path.join(model_dir, epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ensembles.to_csv(os.path.join(output_dir, f'{ensemble_name}_predicts.csv'))
        ens_stats.to_csv(os.path.join(output_dir, f'{ensemble_name}_stats.csv'))
        
    ensembles, uncertainty = form_ensembles(model_dir, models, epoch)
    ens_stats = gen_stats(ensembles)
    if model_dir and ensemble_name:
        save_ensemble(ensembles, ens_stats, model_dir, epoch)
    if ensembles.index.nlevels > 1:
        save_source_stats(models, ensembles, list(ensembles.columns.drop('y')), ensemble_name,
                          model_dir, epoch_key=epoch, target_column='y')
    if isinstance(models, ModelList):
        if epoch:
            models.epoch_test_predicts[epoch] = ensembles
            models.epoch_test_stats[epoch] = ens_stats
        else:
            models.test_predicts = ensembles
            models.test_stats = ens_stats
    else:
        return ensembles, uncertainty, ens_stats


def generate_ensembles(model_predicts, ensemble_runs, ensemble_sizes, classify=False,
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
    classify: bool, optional
      - True: The labels and predictions are classes
      - False: The labels and predictions are numbers. The default is
        False.
    precision : int, optional
        The precision of the ensemble statistics. The default is 2.
    uncertainty : bool, optional
      - True: Return the standard deviations of the individual
        predictions as an uncertainty estimate
      - False: Do not return uncertainty estimates. The default False.
    random_seed : int, optional
        The random number generator seed. The default is 99.

    Returns
    -------
    preds : list
        a list of the predictions made by each ensemble. A list of
        lists if model_predicts is a list of data frames.
    std_devs : list
        The standard deviations of the individual predictions (the
        uncertainty estimate). Only returned if ``uncertainty`` is True.
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
                s = {p[0]: calc_statistics(pred.y, p[1], classify=classify, precision=precision)
                     for p in p_iter}
                stats.append(pd.DataFrame.from_dict(s, orient='index'))
        else:
            for run in range(ensemble_runs):
                if run % 10 == 9:
                    print(run+1, end='')
                else:
                    print('.', end='')
                sample_list = random.sample(range(num_models), k=ensemble_size)
                pred_list = [model_predicts[s] for s in sample_list]
                ensemble_ = create_ensembles(None, pred_list, classify=classify, ensemble_name='',
                                             precision=precision)
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
