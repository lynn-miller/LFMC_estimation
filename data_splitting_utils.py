"""Data Splitting utilities"""

import itertools
import json
import numpy as np
import os
import pandas as pd
import random
import warnings

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from data_prep_utils import normalise

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


def _samples_filter(parent_filter, samples, X, parent_results, parent_incl):
    samp_idx = samples.index
    num_samp = samples.shape[0]
    if parent_incl:  # Add the parent results to the model inputs 
        parent_data = parent_results.reindex(samp_idx).to_numpy().reshape(num_samp, -1)
        if isinstance(parent_incl, dict):
            parent_data = normalise(parent_data, **parent_incl)
        X['parent'] = parent_data
        print(f"parent shape: {X['parent'].shape}")

    if parent_filter:  # Create a filter index based on the parent results
        filter_method = parent_filter[0]
        filter_params = parent_filter[1:]
        filter_str = ', '.join([str(x) for x in filter_params])
        print(f"Samples filtered on parent results using \"{filter_method}({filter_str})\"")
        filter_ = getattr(parent_results, filter_method)(*filter_params)
        return filter_.reindex(samp_idx).to_numpy()

    else:  # No filtering based on parent results is required
        return np.array([True] * num_samp)
    
    
def partition_by_year(sample_dates, year, train_index, val_index, test_index, num_years=None,
                      test_all_years=False, train_adjust=0, test_adjust=0):
    """Partitions a dataset by year
    
    Ensures temporal separation of the test and training samples. The
    main usage is to ensure all samples in the training and validation
    sets were collected before ``year``, and all samples in the test
    set were collected on or after ``year``. Each input index is a bool
    index listing the candidate samples. The function sets the index to
    `False` for samples that were not collected in the right years.
    
    NOTES:
      1. `train_adjust` and/or `test_adjust` can be used to adjust\
        the start/end of the training and/or test sets from the\
        beginning/end of the year.
      2. If `test_adjust` < `-train_adjust`, some samples may be in\
        both the training and test sets.
      3. If `train_adjust` is `"nonTest"`, the training and validation\
        date range is all dates outside the test date range.
      4. The validation set is adjusted in the same way as the\
        training set.

    Parameters
    ----------
    sample_dates : pd.Series
        The sampling date for each sample.
    year : int
        The first year to include in the test set.
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
    num_years : int, optional
        The number of years to include in the test set. Samples
        collected in years later than `year + num_years` are discarded.
        Specify `None` or `0` to include all data from or after `year`
        in the test set.
    test_all_years : bool, optional
        If ``True``, the test index is returned unchanged (train_index
        and val_index are still updated). The default is False.
    train_adjust : int or str, optional
        If int: The number of days to adjust the end date of the training and
        validation sets. E.g. `90` will remove all samples within 90
        days of the end of the training set. These samples are
        discarded (NOT added to the test set).
        If str: The only valid value is 'nonTest'. Only the samples
        that are in the test set are removed from the training and
        validation sets. 
        The default is 0.
    test_adjust : int or str, optional
      - If `int`: The number of days to adjust the start and end dates
        of the test set. A postive number will shift the dates forward
        and a negative number will shift them backwards. E.g. setting
        `test_adjust` to `90`, `year` to `2014` and `num_year` to `1`
        will result in a test set containing samples from `01-Apr-2014`
        to `31-Mar-2015`. Samples in the adjustment period (e.g.
        `01-Jan-2014` to `31-Mar-2014`) are discarded (NOT added to the
        training or validation sets).
      - If `str`: The only valid `str` value is `nonTest`. All current
        training samples not in the test date range will be retained.
      - The default is 0.

    Returns
    -------
    train_index : np.array
        The updated training index with all samples for the test years
        set to False.
    val_index : np.array
        The updated validation index with all samples for the test
        years set to False.
    test_index : np.array
        The updated test index with all samples for years other than
        the test years set to False.

    """
    temp_dates = pd.to_datetime(sample_dates.astype(str))
    train_end = pd.to_datetime(year, format='%Y')
    if test_index is not None and not test_all_years:
        test_adjust = test_adjust or 0
        test_start = pd.to_datetime(year, format='%Y') + pd.Timedelta(test_adjust, 'D')
        if num_years:
            test_end = pd.to_datetime(year + num_years, format='%Y') \
                + pd.Timedelta(test_adjust, 'D')
        else:
            test_end = pd.Timestamp.max
        adjust_index = temp_dates.between(test_start, test_end, inclusive='left')
        test_index = test_index & adjust_index
    if isinstance(train_adjust, int):
        if train_adjust:
            train_end = train_end - pd.Timedelta(train_adjust, 'D')
        adjust_index = temp_dates < train_end
        if test_adjust < -train_adjust:
            warnings.warn(f'Test adjustment ({test_adjust} days) overlaps training adjustment '
                          f'({train_adjust} days) by {-train_adjust - test_adjust} days. Some '
                          'samples may be in both the training and test data.')
    elif train_adjust == 'nonTest':
        adjust_index = ~ adjust_index
    else:
        raise ValueError(f'Invalid train_adjust value: {train_adjust}. ')
    train_index = train_index & adjust_index
    if val_index is not None:
        val_index = val_index & adjust_index
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
            train[i] = idx[pos]
            val[i] = not idx[pos]
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


def kfold_indexes(data, n_folds, test_folds=1, split_col='Site', stratify=None,
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
    test_folds : int
        The number of folds required in each test set. If greater than
        1, all combinations of folds of the required size are generated
        for the test sets, with the remaining data in the train and
        validation sets. Consequntly, the number of train/test splits
        generated is ``<n_folds>C<test_folds>``.
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
    def multi_folds(folds, test_folds):
        new_folds = []
        for x in itertools.combinations(range(len(folds)), test_folds):
            test_ = folds[x[0]]['test']
            train_ = folds[x[0]]['train']
            for y in x:
                test_ = np.union1d(test_ , folds[y]['test'])
                train_ = np.setdiff1d(train_, folds[y]['test'])
            new_folds.append({'train': train_, 'test': test_})
        return new_folds
        
        
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
    
    # Create the folds
    sss = Splitter(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds = [{'train': i1, 'test': i2} for i1, i2 in sss.split(y, y)]
    if test_folds > 1:
        folds = multi_folds(folds, test_folds)
        n_folds = len(folds)
    
    # Generate the test indexes
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


def split_data(data, train_index, test_index, val_index=None,
               input_name='', model_dir='', save_params=False, **normal):
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
    train_data = data[train_index]
    val_data = data[val_index] if val_index is not None else None
    test_data = data[test_index] if test_index is not None else None
    return train_data, test_data, val_data


def train_test_split(model_params, samples, X, y, parent_results):
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
    
    splitColumn:
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
    def get_split_indexes(model_params, samples):
        if model_params['splitMethod'] == 'random':
            train_index, val_index, test_index = random_split(
                    num_samples=samples.shape[0],
                    split_sizes=model_params['splitSizes'],
                    val_data=model_params['validationSet'],
                    random_seed=model_params['randomSeed'])
        elif model_params['splitMethod'] == 'bySite':
            train_index, val_index, test_index = split_by_site(
                    data=samples,
                    split_sizes=model_params['splitSizes'],
                    split_col=model_params['splitColumn'],
                    stratify=model_params['splitStratify'],
                    val_data=model_params['validationSet'],
                    random_seed=model_params['randomSeed'])
        if model_params['splitYear']:
            train_index, val_index, test_index = partition_by_year(
                samples[model_params['yearColumn']],
                model_params['splitYear'],
                train_index, val_index, test_index,
                test_all_years=model_params['testAllYears'],
                train_adjust=model_params['trainAdjust'],
                test_adjust=model_params['testAdjust'])
        return train_index, val_index, test_index

    def get_yearly_indexes(model_params, samples):
        full_index = pd.Series([True] * samples.shape[0], index=samples.index)
        train_index, val_index, test_index = partition_by_year(
            samples[model_params['yearColumn']],
            model_params['splitYear'],
            full_index, None, full_index,
            num_years=model_params['splitFolds'],
            test_all_years=model_params['testAllYears'],
            train_adjust=model_params['trainAdjust'],
            test_adjust=model_params['testAdjust'])
        if model_params['validationSet']:
            train_index, val_index = train_val_split(
                    num_samples=samples.shape[0],
                    val_size=model_params['splitSizes'][1],
                    train_val_idx=train_index,
                    random_seed=model_params['randomSeed'])
        return train_index, val_index, test_index

    def get_nosplit_indexes(model_params, samples):        
        test_index = None
        train_index = np.array([True] * samples.shape[0])
        if model_params['validationSet']:
            train_index, val_index = train_val_split(
                    num_samples=samples.shape[0],
                    val_size=model_params['splitSizes'][1],
                    train_val_idx=train_index,
                    random_seed=model_params['randomSeed'])
        else:
            val_index = None
        return train_index, val_index, test_index

    if model_params['splitMethod'] in ['random', 'bySite']:
        train_index, val_index, test_index = get_split_indexes(model_params, samples)
    elif model_params['splitMethod'] == 'byYear':
        train_index, val_index, test_index = get_yearly_indexes(model_params, samples)
    elif not model_params['splitMethod']:
        train_index, val_index, test_index = get_nosplit_indexes(model_params, samples)
    else:
        raise ValueError(f"Invalid train/test split method: {model_params['splitMethod']}")

    if parent_results is not None:
        parent_filter = model_params['parentFilter']
        parent_incl = model_params['parentResult']
        samples_index = _samples_filter(parent_filter, samples, X, parent_results.base, parent_incl)
        train_index = train_index & samples_index
        test_index = None if test_index is None else (test_index & samples_index)
        val_index = None if val_index is None else (val_index & samples_index)


    model_dir = model_params['modelDir']
    save_params = model_params['saveModels']
    normal = model_params['targetNormalise'] or {}
    y_train, y_test, y_val = split_data(y, train_index, test_index, val_index,
                                        'target', model_dir, save_params, **normal)
    data = {'train': {'X': {}, 'y': y_train},
            'val':   {'X': {}, 'y': y_val},
            'test':  {'X': {}, 'y': y_test}}
    for source, xdata in X.items():
        if source == 'aux':
            normal = {}
        else:
            normal = model_params['inputs'][source].get('normalise', {}) or {}
        train, test, val = split_data(xdata, train_index, test_index, val_index,
                                      source, model_dir, save_params, **normal)
        data['train']['X'][source] = train
        data['val']['X'][source] = val
        data['test']['X'][source] = test
    return data


def kfold_split(model_params, samples, X, y, parent_results):
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
    
    splitColumn:
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
    def get_fold_indexes(model_params, samples):
#        folds = range(model_params['splitFolds'])
        val_size = model_params['splitSizes'][1] if model_params['validationSet'] else 0
        if model_params['splitMethod'] == 'random':
            train_index, val_index, test_index = kfold_indexes(
                    data=samples.reset_index(),
                    n_folds=model_params['splitFolds'],
                    test_folds=model_params['testFolds'],
                    split_col=samples.index.name or 'index',
                    stratify=False,
                    val_size=val_size,
                    random_seed=model_params['randomSeed'])
        else:   # model_params['splitMethod'] == 'bySite'
            train_index, val_index, test_index = kfold_indexes(
                    data=samples,
                    n_folds=model_params['splitFolds'],
                    test_folds=model_params['testFolds'],
                    split_col=model_params['splitColumn'],
                    stratify=model_params['splitStratify'],
                    val_size=val_size,
                    random_seed=model_params['randomSeed'])
        folds = range(len(train_index))
        if model_params['splitYear']:
            year_folds = model_params.get('yearFolds', 0)
            if year_folds <= 1:
                for n in folds:
                    train_index[n], val_index[n], test_index[n] = partition_by_year(
                        samples[model_params['yearColumn']],
                        model_params['splitYear'],
                        train_index[n], val_index[n], test_index[n],
                        num_years=year_folds,
                        test_all_years=model_params['testAllYears'],
                        train_adjust=model_params['trainAdjust'],
                        test_adjust=model_params['testAdjust'])
            else:
                first_year = model_params['splitYear']
                last_year = first_year + year_folds
                year_folds = list(reversed(range(first_year, last_year)))
                trn_ind = {}
                val_ind = {}
                tst_ind = {}
                new_folds = []
                for n in folds:
                    for year in year_folds:
                        fold = f'{year}_{n:02d}'
                        new_folds.append(fold)
                        trn_ind[fold], val_ind[fold], tst_ind[fold] = partition_by_year(
                            samples[model_params['yearColumn']],
                            year,
                            train_index[n], val_index[n], test_index[n],
                            num_years=1,
                            test_all_years=model_params['testAllYears'],
                            train_adjust=model_params['trainAdjust'],
                            test_adjust=model_params['testAdjust'])
                folds = new_folds
                train_index = trn_ind
                val_index = val_ind
                test_index = tst_ind
        return folds, train_index, val_index, test_index
    
    def get_yearly_indexes(model_params, samples):
        train_index = {}
        val_index = {}
        test_index = {}
        first_year = model_params['splitYear']
        last_year = first_year + model_params['yearFolds']  # model_params['splitFolds']
        folds = list(reversed(range(first_year, last_year)))
        val_size = model_params['splitSizes'][1] if model_params['validationSet'] else 0
        for year in range(first_year, last_year):
            full_index = pd.Series([True] * samples.shape[0], index=samples.index)
            train_index[year], val_index[year], test_index[year] = partition_by_year(
                samples[model_params['yearColumn']],
                year, full_index, None, full_index,
                num_years=1,
                test_all_years=model_params['testAllYears'],
                train_adjust=model_params['trainAdjust'],
                test_adjust=model_params['testAdjust'])
            if val_size > 0:
                train_index[year], val_index[year] = train_val_split(
                        num_samples=samples.shape[0],
                        val_size=val_size,
                        train_val_idx=train_index,
                        random_seed=model_params['randomSeed'])
        return folds, train_index, val_index, test_index
        
    def save_folds(model_params, samples, train_index, val_index, test_index):
        kfold_split.folds = folds
        kfold_split.indexes = {'train': train_index, 'val': val_index, 'test': test_index}
        if model_params.get('run', None) is None or model_params['resplit']:
            save_dir = model_params['modelDir']
        else:
            save_dir = os.path.dirname(model_params['modelDir'].rstrip('\\/'))
        with open(os.path.join(save_dir, 'test_folds.json'), 'w') as f:
            save_index = test_index.copy()
            for fold in folds:
                save_index[fold] = samples.index[save_index[fold]].tolist()
            json.dump(save_index, f, indent=2)
    
    def get_fold_data(fold, X, y, train_index, val_index, test_index, samples_index, model_params):
        train_index = train_index[fold] & samples_index
        test_index = None if test_index[fold] is None else (test_index[fold] & samples_index)
        val_index = None if val_index[fold] is None else (val_index[fold] & samples_index)
        model_dir = model_params['modelDir']
        save_params = model_params['saveModels']
        normal = model_params['targetNormalise'] or {}
        y_train, y_test, y_val = split_data(y, train_index, test_index, val_index,
                                            'target', model_dir, save_params, **normal)
        data = {'train': {'X': {}, 'y': y_train},
                'val':   {'X': {}, 'y': y_val},
                'test':  {'X': {}, 'y': y_test}}
        for source, xdata in X.items():
            if source == 'aux':
                normal = {}
            else:
                normal = model_params['inputs'][source].get('normalise', {}) or {}
            train, test, val = split_data(xdata, train_index, test_index, val_index,
                                          source, model_dir, save_params, **normal)
            data['train']['X'][source] = train
            data['val']['X'][source] = val
            data['test']['X'][source] = test
        return data

    if not kfold_split.folds or model_params['resplit']:
        if model_params['splitMethod'] in ['random', 'bySite']:
            folds, train_index, val_index, test_index = get_fold_indexes(model_params, samples)
        elif model_params['splitMethod'] == 'byYear':
            model_params['yearFolds'] = model_params.get('yearFolds', 0) \
                                        or model_params['splitFolds']
            folds, train_index, val_index, test_index = get_yearly_indexes(model_params, samples)
        elif not model_params['splitMethod']:
            raise ValueError(
                'Invalid train/test split: "splitFolds" > 1 specified with no "splitMethod"')
        else:
            raise ValueError(f"Invalid train/test split method: {model_params['splitMethod']}")
        save_folds(model_params, samples, train_index, val_index, test_index)

    else:
        folds = kfold_split.folds
        train_index = kfold_split.indexes['train']
        val_index = kfold_split.indexes['val']
        test_index = kfold_split.indexes['test']

    get_samples_index = False
    if parent_results is None:
        samples_index = np.array([True] * samples.shape[0])
    else:
        parent_filter = model_params['parentFilter']
        parent_incl = model_params['parentResult']
        try:
            parent_data = parent_results.base
            samples_index = _samples_filter(parent_filter, samples, X, parent_data, parent_incl)
        except:
            get_samples_index = True

    for fold in folds:
        if get_samples_index:
            year = fold.split('_')[0]
            parent_data = parent_results[f'{year}_base']
            samples_index = _samples_filter(parent_filter, samples, X, parent_data, parent_incl)
        data = get_fold_data(fold, X, y, train_index, val_index,
                             test_index, samples_index, model_params)
        yield (fold, data)

# Add attributes to kfold_split to store the fold indexes between runs
kfold_split.indexes = None
kfold_split.folds = None
