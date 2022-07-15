"""Model building and evaluation utilities"""

import glob
import numpy as np
import os
import pandas as pd
import random

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from model_list import ModelList
from analysis_utils import calc_statistics
from lfmc_model import import_model_class


# =============================================================================
# Data splitting functions
# =============================================================================


def _stratify_sites(data, split_col, stratify, min_sites):
    print(f'Split by {split_col}, stratify by {stratify}')
    # Assign a unique landcover value to each site
    sites = data.groupby([split_col], as_index=False).agg(
        stratify=(stratify, lambda x: pd.Series.mode(x)[0]))
    lc = sites.groupby(['stratify'], as_index=False).agg(counts=(split_col, 'count'))
    # Group landcover classes with fewer than min_sites sites into one group
    lc = lc.stratify[lc.counts < min_sites]
    if pd.api.types.is_numeric_dtype(sites.stratify):  # Stratify values are numeric
        dummy_value = -99
    else:  # Stratify values are strings
        dummy_value = '@#$%'
    sites.loc[sites.stratify.isin(lc), 'stratify'] = dummy_value
    # Return the stratified sites
    return sites
    
    
def partition_by_year(sample_years, year, train_index, val_index, test_index):
    """Partitions a dataset by year
    
    Ensures all samples in the training and validation sets were
    collected before ``year``, and all samples in the test set were
    collected on or after ``year``. Each input index is a bool index
    listing the candidate samples. The function sets the index to False
    for samples that were not collected in the right years.  
    

    Parameters
    ----------
    sample_years : pd.Series
        The sampling year for each sample.
    year : int
        The first year to include in the test set. All later years will
        also be in the test test.
    train_index : np.array
        1-dimensional bool array the same length as sample_years.
        Samples in the training set should be set to True, all others
        to False.
    val_index : np.array
        1-dimensional bool array the same length as sample_years.
        Samples in the validation set should be set to True, all others
        to False.
    test_index : np.array
        1-dimensional bool array the same length as sample_years.
        Samples in the test set should be set to True, all others
        to False.

    Returns
    -------
    train_index : np.array
        The updated training index with all samples for the test years
        set to False.
    val_index : TYPE
        The updated validation index with all samples for the test
        years set to False.
    test_index : TYPE
        The updated test index with all samples for years other than
        the test years set to False.

    """
    year_index = sample_years >= year
    train_index = train_index & (~ year_index)
    if val_index is not None:
        val_index = val_index & (~ year_index)
    if test_index is not None:
        test_index = test_index & year_index
    return train_index, val_index, test_index

	
def random_split(num_samples, split_sizes, val_data=False, random_seed=None):
    """Splits samples randomly
    
    Randomly assigns numbers in range(0, num_samples) to the test,
    training and validation set in accordance with the required split
    sizes. Returns an index for each set that indicates which samples
    are in each set.

    Parameters
    ----------
    num_samples : int
        The number of samples.
    split_sizes : tuple of length 1 or 2
        The size of the test and optionally the validation set. If the
        tuples are integers, they are interpreted as the number of
        samples. If they are floats, they are interpreted as the
        proportion of samples.
    val_data : bool, optional
        Indicates if validation data is required. The default is False.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the training set, false otherwise.
    val_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the validation set, false otherwise.
    test_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the test set, false otherwise.
    """
    if random_seed is not None:
        random.seed(random_seed)
    sample_list = list(range(num_samples))
    random.shuffle(sample_list)

    # Generate the test index
    test_size = split_sizes[0]
    if type(test_size) is int:
        break1 = test_size
    else:
        break1 = int(np.floor(num_samples * test_size))
    test_index = np.zeros(num_samples, dtype=bool)
    test_index[sample_list[:break1]] = True

    # Generate the validation index if used
    if val_data:
        val_size = split_sizes[1]
        if type(val_size) is int:
            break2 = break1 + val_size
        else:
            break2 = break1 + int(np.floor(num_samples * (val_size)))
        val_index = np.zeros(num_samples, dtype=bool)
        val_index[sample_list[break1:break2]] = True
    else:
        break2 = break1
        val_index = None

    # Generate the training index
    train_index = np.zeros(num_samples, dtype=bool)
    train_index[sample_list[break2:]] = True
    return train_index, val_index, test_index


def train_val_split(num_samples, val_size, train_val_idx, random_seed=None):
    """Creates a validation set from the training data.
    

    Parameters
    ----------
    num_samples : int
        Total number of samples (including test samples).
    val_size : int or float
        The number (int) or proportion of the training set (float) to
        allocate to the validation set.
    train_val_idx : np.array
        An array of length num_samples. True for samples in the
        training/validation set, False for samples in the test set.
    random_seed : int, optional
        The random seed. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the training set, false otherwise.
    val_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the validation set, false otherwise.

    """
    idx, _, _ = random_split(train_val_idx.sum(), (0, val_size), True, random_seed)
    train = np.array([False] * num_samples)
    val = np.array([False] * num_samples)
    pos = 0
    for i, x in enumerate(train_val_idx):
        if x:
            if idx[pos]:
                train[i] = True
            else:
                val[i] = True
            pos += 1
    return train, val


def split_by_site(data, split_sizes, split_col='Site', stratify=None,
                val_data=False, min_sites=6, random_seed=None):
    """Splits sample sites randomly.
    
    Randomly assigns sites to the test, training and validation sets in
    accordance with the required split sizes. If stratified splitting
    is requested, sites are grouped according to the stratified column
    value and split so each group is evenly represented in each set.
    After the sites have been split, samples are assigned to the
    appropriate set based on their site.
    
    Parameters
    ----------
    data : DataFrame
        Data Frame of samples. Must have a column labelled with
        ``split_col``, and one labelled with ``stratify`` if specified.
    split_sizes : tuple of length 1 or 2
        The size of the test and optionally the validation set. If the
        tuples are ints, they are interpreted as the number of sites.
        If floats, they are interpreted as the proportion of sites.
    split_col : str, optional
        The name of the column containing the sample site. The default
        is 'Site'.
    stratify : str or None, optional
        The name of the column to use for stratified splitting. None
        for no stratified splits. The default is None.
    val_data : bool, optional
        Indicates if validation data is required. The default is False.
    min_sites : int, optional
        The minimum number of sites in a stratified group. Groups with
        fewer sites are combined into a single group. The default is 6.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the training set, false otherwise.
    val_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the validation set, false otherwise.
    test_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the test set, false otherwise.
    """
    if stratify:
        sites =_stratify_sites(data, split_col, stratify, min_sites)
        y = sites.stratify
        Splitter = StratifiedShuffleSplit
        
    else:
        print(f'Split by {split_col}, no stratify')
        sites = data.groupby([split_col], as_index=False).agg(counts=(split_col, 'count'))
        y = sites[split_col]
        Splitter = ShuffleSplit

    temp = data.reset_index()[split_col]
    
    # Generate the test index
    test_size = split_sizes[0]
    if type(test_size) is float:
        test_size = int(np.floor(y.size * test_size))
    sss = Splitter(n_splits=1, test_size=test_size, random_state=random_seed)
    trainSites, testSites = next(sss.split(y, y))
    test_index = np.zeros(temp.size, dtype=bool)
    test_index[temp.isin(sites.loc[testSites, split_col])] = True

    # Generate the validation index if used
    if val_data:
        val_size = split_sizes[1]
        if type(val_size) is float:
            val_size = int(np.floor(y.size * val_size))
        y1 = y.iloc[trainSites]
        rs = None if random_seed is None else random_seed * 2
        sss = Splitter(n_splits=1, test_size=val_size, random_state=rs)
        ti, vi = next(sss.split(y1, y1))
        valSites = trainSites[vi]
        trainSites = trainSites[ti]
        val_index = np.zeros(temp.size, dtype=bool)
        val_index[temp.isin(sites.loc[valSites, split_col])] = True
    else:
        val_index = None

    # Generate the training index
    train_index = np.zeros(temp.size, dtype=bool)
    train_index[temp.isin(sites.loc[trainSites, split_col])] = True
    return train_index, val_index, test_index	


def kfold_indexes(data, n_folds, split_col='Site', stratify=None,
                  val_size=0, min_values=6, random_seed=None):
    """Creates k-fold splits
    
    Creates ``nFold`` splits of the "sites" by randomly assigning sites
    to a fold. Then, in turn, each fold is assigned to the test set and
    remaining folds assigned to the training set. (Validation data is
    randomly selected from training data, if required). If stratified
    splitting is requested, sites are grouped by the stratified column
    value and split so each group is evenly represented in each fold.
    After the sites have been split, samples are assigned to the
    appropriate set based on their site.
    
    Note that if split_col is set to the name of the data index (or
    "index" if unnamed), the folds will be random (stratified) splits
    of the samples, rather than the sites.

    Parameters
    ----------
    data : DataFrame
        Data Frame of samples. Must have a column labelled with
        ``split_col``, and one labelled with ``stratify`` if specified.
    n_folds : int
        The number of folds (``K``) to create.
    split_col : str, optional
        The name of the column to use for splitting. The default is
        'Site'. Use the index name for random splitting.
    stratify : str or None, optional
        The name of the column to use for stratified splitting. None
        for no stratified splits. The default is None.
    val_size : int or float, optional
        The number (int) or proportion (float) of samples to assign to
        the validation set in each fold. The validation samples are
        randomly selected from the training set. If 0, no validation
        sets are created. The default is 0.
    min_values : int, optional
        The minimum number of ``split_col`` values in a stratified
        group. Groups with fewer values are combined into one group.
        The default is 6.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : list of bool arrays
        A list containing a bool array for each fold. The arrays are
        of length equal to rows in ``data``. Elements are True if the
        sample should be included in the training set for the fold,
        false otherwise.
    val_index : list of bool arrays
        A list containing a bool array for each fold. The arrays are
        of length equal to rows in ``data``. Elements are True if the
        sample should be included in the validation set for the fold,
        false otherwise.
    test_index : list of bool arrays
        A list containing a bool array for each fold. The arrays are
        of length equal to rows in ``data``. Elements are True if the
        sample should be included in the test set for the fold, false
        otherwise.
    """
    if stratify:
        split_values =_stratify_sites(data, split_col, stratify, min_values)
        y = split_values.stratify
        Splitter = StratifiedKFold
        ValSplitter = StratifiedShuffleSplit
        
    else:
        print(f'Split by {split_col}, no stratify')
        split_values = data.groupby([split_col], as_index=False).agg(counts=(split_col, 'count'))
        y = split_values[split_col]
        Splitter = KFold
        ValSplitter = ShuffleSplit

    temp = data.reset_index()[split_col]
    
    # Generate the test indexes
    sss = Splitter(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds = [{'train': i1, 'test': i2} for i1, i2 in sss.split(y, y)]
    test_index = [np.zeros(temp.size, dtype=bool) for _ in range(n_folds)]
    for n, fold in enumerate(folds):
        test_index[n][temp.isin(split_values.loc[fold['test'], split_col])] = True

    # Generate the validation indexes if used
    if val_size:
        if type(val_size) is float:
            val_size = int(np.floor(y.size * val_size))
        rs = None if random_seed is None else random_seed * 2
        val_index = [np.zeros(temp.size, dtype=bool) for _ in range(n_folds)]
        for n, fold in enumerate(folds):
            y1 = y.iloc[fold['train']]
            sss = ValSplitter(n_splits=1, test_size=val_size, random_state=rs)
            ti, vi = next(sss.split(y1, y1))
            valSites = fold['train'][vi]
            fold['train'] = fold['train'][ti]
            val_index[n][temp.isin(split_values.loc[valSites, split_col])] = True
    else:
        val_index = [None] * n_folds

    # Generate the training indexes
    train_index = [np.zeros(temp.size, dtype=bool) for n in range(n_folds)]
    for n, fold in enumerate(folds):
        train_index[n][temp.isin(split_values.loc[fold['train'], split_col])] = True

    return train_index, val_index, test_index


def split_data(data, train_index, test_index, val_index=None):
    """Splits data into train, test and validation sets
    
    Splits data into train, test and (optionally) validation sets using
    the index parameters.

    Parameters
    ----------
    data : array
        DESCRIPTION.
    train_index : array of bool
        Array values should be True for entries in the training set,
        False otherwise.
    test_index : array of bool
        Array values should be True for entries in the test set, False
        otherwise.
    val_index : array of bool, optional
        Array values should be True for entries in the validation set,
        False otherwise. If val_index is None, no validation set is
        created. The default is None.

    Returns
    -------
    train_data : array
        The training data.
    test_data : array
        The test data.
    val_data : array or None
        The validation data (if any).
    """
    #split the sample set into training, validation and testing sets
    test_data = data[test_index] if test_index is not None else None
    val_data = data[val_index] if val_index is not None else None
    train_data = data[train_index]
    return train_data, test_data, val_data

    
# =============================================================================
# Data preparation functions
# =============================================================================
    

def reshape_data(X, nchannels):
    """Reshape an array
    
    Reshape a two dimensional array of rows with flattened timeseries
    and channels to three dimensions.

    Parameters
    ----------
    X : np.array
        A two dimensional array. The first dimension is the row, the
        second the features ordered by date then band/channel - e.g.
        d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...
    nchannels : int
        Number of channels.

    Returns
    -------
    np.array
        A 3 dimensional array where the first dimension are rows, the
        second time and the third the band or channel.
    """
    return X.reshape(X.shape[0], int(X.shape[1] / nchannels), nchannels)


def get_bounds(data, percentiles):
    """Gets the percentile bounds from an array
    

    Parameters
    ----------
    data : np.array
        Input dataset. Must have at least two dimensions.
    percentiles : int or tuple
        Percentiles for bounds. If int, this is assumed to be the lower
        percentile, and the upper percentile is 100 - lower percentile.
        If tuple-like, it should be the lower and upper percentile.

    Returns
    -------
    np.array
        The bounds, calculated across the first two dimensions.

    """
    if type(percentiles) is int:
        percentiles = [percentiles, 100-percentiles]
    return np.percentile(data, percentiles, axis=(0, 1))
    
    
def normalise(data, method='minMax', percentiles=0, range=[0, 10000], out_range=(0, 1)):
    """Normalises the data.
    
    Parameters
    ----------
    data : array
        Array of the data to be normalised.
    method : str, optional
        Normalisation method to use. Currently implemented methods are:
          - minMax: Default. Normalise so the lower percentile is 0 and
            the upper percentile is 1.
          - range: Normal across a set range, so lower range is 0 and
            upper range is 1.
    percentiles : int or list-like, optional
        The percentile to use with the minMax method, If a single int
        value, this is treated as the lower percentile and the upper
        percentile is 100 - the value. The default is 0.
    range : tuple or list-like, optional
        Two values corresponding to the lower and upper bounds of the
        normalisation range. The default is [0, 10000].
    out_range : tuple or list-like, optional
        Two values corresponding to the lower and upper bounds of the
        output range. After normalising, the data is re-scaled to this
        range. The default is (0, 1).

    Returns
    -------
    array
        The normalised data.
    """
    if method == 'minMax':
        bounds = get_bounds(data, percentiles)
        temp = (data - bounds[0]) / (bounds[1] - bounds[0])
    elif method == 'range':
        temp = (data - range[0]) / (range[1] - range[0])
    else: # method not implemented
        raise ValueError(f'Invalid method: {method}')
    if tuple(out_range) == (0, 1):
        return temp
    else:
        return temp * (out_range[1] - out_range[0]) + out_range[0]
    

# =============================================================================
# Process test results functions
# =============================================================================


def merge_kfold_results(model_dir, models=None, folds=None, epochs=False):
    """Merges fold predictions
    
    Merges the predictions from each fold to create a predictions file
    for the run. If results from intermediate epochs have been saved
    then the epoch predictions are also merged. Statistics for each
    merged set of predictions are saved.

    Parameters
    ----------
    model_dir : str
        Name of directory containing the folds, and where the merged
        results will be written.
    models : ModelList, optional
        The kfold models to merge. The model results should be stored
        in the ``all_results`` attribute on each model. If ``None`` the
        model results will be loaded from ``model_dir``. The default is
        None.
    folds : list, optional
        List of fold names. If ``None`` fold names are derived from any
        sub-directories in ``model_dir`` that start with ``fold``. The
        default is None.
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
        predictions = pd.concat(predictions)
        stats = {}
        for y in predictions.columns.drop('y'):
            stats[y] = calc_statistics(predictions.y, predictions[y])
        all_stats = pd.DataFrame.from_dict(stats, orient='index')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        predictions.to_csv(os.path.join(output_dir, 'predictions.csv'))
        all_stats.to_csv(os.path.join(output_dir, 'predict_stats.csv'))
        return predictions, all_stats
    
    def load_predictions(model_dir, folds, epoch=''):
        predictions = []
        for fold in folds:
            pred_file = os.path.join(model_dir, fold, epoch, 'predictions.csv')
            predictions.append(pd.read_csv(pred_file, index_col=0))
        return predictions
        
    if not folds:
        folds = [os.path.basename(dir_) for dir_ in glob.glob(os.path.join(model_dir, 'fold*'))]
    if epochs:
        epoch_dirs = glob.glob(os.path.join(model_dir, 'fold*', 'epoch*'))
        epoch_dirs = sorted({os.path.basename(dir_) for dir_ in epoch_dirs})
        for epoch in epoch_dirs:
            predictions = load_predictions(model_dir, folds, epoch)
            _ = save_results(predictions, os.path.join(model_dir, epoch))
    if models:
        predictions = [model.all_results for model in models]
    else:
        predictions = load_predictions(model_dir, folds)
    all_results, all_stats = save_results(predictions, model_dir)
    return all_results, all_stats


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
        stack_stats = pd.concat([model.all_stats.stack() for model in models], axis=1)
        output_dir = model_dir
    else:
        output_dir = os.path.join(model_dir, epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        stats_list = glob.glob(os.path.join(model_dir, 'run*', epoch, 'predict_stats.csv'))
        stack_stats = [pd.read_csv(stats_file, index_col=0).stack() for stats_file in stats_list]
        stack_stats = pd.concat(stack_stats, axis=1)
    # Calculate mean and variances of the run prediction statistics
    stack_stats.to_csv(os.path.join(output_dir, 'stats_all.csv'))
    means = stack_stats.mean(axis=1).unstack()
    means.to_csv(os.path.join(output_dir, 'stats_means.csv'), float_format="%.2f")
    variances = stack_stats.var(axis=1).unstack()
    variances.to_csv(os.path.join(output_dir, 'stats_vars.csv'), float_format="%.2f")
    if models:
        models.run_stats = stack_stats
        models.means = means
        models.variances = variances


def create_ensembles(model_dir, models='run*', epoch='', ensemble_name='ensemble'):
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
        a ModelList or a list of predictions and ensembles will not be
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
    if isinstance(models, ModelList):
        stack_results = pd.concat([model.all_results.stack() for model in models], axis=1)
        output_dir = model_dir
    elif model_dir is None:
        stack_results = pd.concat([model.stack() for model in models], axis=1)
    else:
        output_dir = os.path.join(model_dir, epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if models is None:
            preds_list = glob.glob(os.path.join(model_dir, 'run*', epoch, 'predictions.csv'))
        elif isinstance(models, str):
            preds_list = glob.glob(os.path.join(model_dir, models, epoch, 'predictions.csv'))
        else:
            preds_list = []
            for fn in models:
                preds_list.extend(glob.glob(os.path.join(model_dir, fn, epoch, 'predictions.csv')))
        stack_results = [pd.read_csv(preds_file, index_col=0).stack() for preds_file in preds_list]
        stack_results = pd.concat(stack_results, axis=1)
    ensembles = stack_results.mean(axis=1).unstack()
    stats = {}
    for y in ensembles.columns.drop('y'):
        stats[y] = calc_statistics(ensembles.y, ensembles[y])
    all_stats = pd.DataFrame.from_dict(stats, orient='index')
    if model_dir and ensemble_name:
        ensembles.to_csv(os.path.join(output_dir, f'{ensemble_name}_preds.csv'))
        all_stats.to_csv(os.path.join(output_dir, f'{ensemble_name}_stats.csv'))
    if isinstance(models, ModelList):
        models.all_results = ensembles
        models.all_stats = all_stats
    else:
        return ensembles, all_stats


def generate_ensembles(model_predicts, ensemble_runs, ensemble_sizes, random_seed=99):
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
        stats = []
        num_models = len(model_predicts)
        if ensemble_size == 1:
            if ensemble_runs < num_models:
                sample_list = random.sample(range(num_models), k=ensemble_runs)
                for sample in sample_list:
                    preds.append(model_predicts[sample])
            else:
                preds = model_predicts
            for p in preds:
                p_iter = p.drop('y', axis=1).iteritems()
                s = {pred_[0]: calc_statistics(p.y, pred_[1]) for pred_ in p_iter}
                stats.append(pd.DataFrame.from_dict(s, orient='index'))
        else:
            for run in range(ensemble_runs):
                if run % 10 == 9:
                    print(run+1, end='')
                else:
                    print('.', end='')
                sample_list = random.sample(range(num_models), k=ensemble_size)
                pred_list = [model_predicts[s] for s in sample_list]
                ensemble_ = create_ensembles(None, pred_list, ensemble_name='')
                preds.append(ensemble_[0])
                stats.append(ensemble_[1])
        return preds, stats

    random.seed(random_seed)
    all_stats = []
    predict = []
    if isinstance(model_predicts[0], pd.DataFrame):
        # Generate ensembles of varying sizes
        if isinstance(ensemble_sizes, int):
            ensemble_sizes = [ensemble_sizes]
        for ensemble_size in ensemble_sizes:
            print(f'Generating ensembles - size {ensemble_size}:', end=' ')
            preds_, stats_ = gen_ensemble_set(
                model_predicts, ensemble_runs, ensemble_size)
            all_stats.append(stats_)
            predict.append(preds_)
            print('')
    else:
        # Generate fixed-size ensembles for each test
        if isinstance(ensemble_sizes, list):
            # Limited to one ensemble size if more than one test
            ensemble_sizes = ensemble_sizes[0]
        for test_ in range(len(model_predicts)):
            print(f'Generating ensembles - test {test_}:', end=' ')
            preds_, stats_ = gen_ensemble_set(
                model_predicts[test_], ensemble_runs, ensemble_sizes)
            all_stats.append(stats_)
            predict.append(preds_)
            print('')
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
    derived_models = model.params.get('derivedModels', False)
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
    else:
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
        results[model_name] = model.evaluate(data['X'], data['y'], model_name, plot=False)

    # Create dataframes for predictions and stats
    all_results = pd.DataFrame({'y': data['y'],
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
        model.all_results = all_results
        model.all_stats = all_stats
    else:
        preds_file = f'{test_name}_predicts.csv'
        stats_file = f'{test_name}_stats.csv'

    # Save results to CSV files
    if save_preds:
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
    model_dir = os.path.join(model_params['modelDir'], '')
    model = import_model_class(model_params['modelClass'])(model_params, inputs=train['X'])
    try:
        if model_params.get('evaluateEpochs'):
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
