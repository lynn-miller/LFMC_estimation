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


def _stratify_split(data, split_col, stratify, min_values):
    print(f'Split by {split_col}, stratify by {stratify}')
    # Assign a unique stratify value to each site
    groups = data.groupby([split_col], as_index=False).agg(
        stratify=(stratify, lambda x: pd.Series.mode(x)[0]))
    strat = groups.groupby(['stratify'], as_index=False).agg(counts=(split_col, 'count'))
    # Group stratify classes with fewer than min_sites sites into one group
    strat = strat.stratify[strat.counts < min_values]
    if pd.api.types.is_numeric_dtype(groups.stratify):  # Stratify values are numeric
        dummy_value = -99
    else:  # Stratify values are strings
        dummy_value = '@#$%'
    groups.loc[groups.stratify.isin(strat), 'stratify'] = dummy_value
    # Return the stratified sites
    return groups


def _samples_filter(parent_filter, samples, X, parent_results): #, parent_normal):
    samp_idx = samples.index
    num_samp = samples.shape[0]
    X['parent'] = parent_results.reindex(samp_idx).to_numpy().reshape(num_samp, -1)
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
    
    
def partition_by_year(sample_dates, year, train_index, test_index, num_years=None,
                      sample_values=None, test_all_years=False, train_adjust=0, test_adjust=0):
    """Partitions a dataset by year
    
    Ensures temporal separation of the test and training samples. The
    main usage is to ensure all samples in the training
    set were collected before ``year``, and all samples in the test
    set were collected on or after ``year``. Each input index is a bool
    index listing the candidate samples. The function sets the index to
    `False` for samples that were not collected in the right years.
    
    NOTES:
      1. `train_adjust` and/or `test_adjust` can be used to adjust\
        the start/end of the training and/or test sets from the\
        beginning/end of the year.
      2. If `test_adjust` < `-train_adjust`, some samples may be in\
        both the training and test sets.
      3. If `train_adjust` is `"nonTest"`, the training and\
        date range is all dates outside the test date range.

    Parameters
    ----------
    sample_dates : pd.Series
        The sampling date for each sample.
    year : int
        The first year to include in the test set.
    train_index : np.array
        1-dimensional bool array the same length as sample_dates.
        Samples in the training set should be set to True, all others
        to False.
    test_index : np.array
        1-dimensional bool array the same length as sample_dates.
        Samples in the test set should be set to True, all others
        to False.
    num_years : int, optional
        The number of years to include in the test set. Samples
        collected in years later than `year + num_years` are discarded.
        Specify `None` or `0` to include all data from or after `year`
        in the test set.
    test_all_years : bool, optional
        If ``True``, the test index is returned unchanged (train_index
        is still updated). The default is False.
    train_adjust : int or str, optional
        If int: The number of days to adjust the end date of the
        training and validation sets. E.g. `90` will remove all samples
        within 90 days of the end of the training set. These samples
        are discarded (NOT added to the test set).
        If str: The only valid value is 'nonTest'. Only the samples
        that are in the test set are removed from the training and
        validation sets. 
        The default is 0.
    test_adjust : int, optional
      - The number of days to adjust the start and end date of the test
        set. A postive number will shift the dates forward and a
        negative number will shift them backwards. E.g. setting
        `test_adjust` to `90`, `year` to `2014` and `num_year` to `1`
        will result in a test set containing samples from `01-Apr-2014`
        to `31-Mar-2015`. Samples in the adjustment period (e.g.
        `01-Jan-2014` to `31-Mar-2014`) are discarded (NOT added to the
        training or validation sets).
      - The default is 0.

    Returns
    -------
    train_index : np.array
        The updated training index with all samples for the test years
        set to False.
    test_index : np.array
        The updated test index with all samples for years other than
        the test years set to False.

    """
    temp_dates = pd.to_datetime(sample_dates.astype(str))
    train_end = pd.to_datetime(year, format='%Y')
    
    # Set up the test index for the test dates
    if test_index is not None and not test_all_years:
        # Create the test adjustment index
        test_adjust = test_adjust or 0
        test_start = pd.to_datetime(year, format='%Y') + pd.Timedelta(test_adjust, 'D')
        if num_years:
            test_end = pd.to_datetime(year + num_years, format='%Y') \
                + pd.Timedelta(test_adjust, 'D')
        else:
            test_end = pd.Timestamp.max
        adjust_index = temp_dates.between(test_start, test_end, inclusive='left')
        # Remove samples outside the test date range from the text index
        test_index = test_index & adjust_index

    # Add any samples with a sample_value not in the test set to the training set
    if sample_values is not None:
        sample_index = ~ sample_values.isin(sample_values[test_index])
        train_index = train_index | sample_index
        
    # Set up the training and validation indexes for the training dates
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
    return train_index, test_index
    
    
def partition_by_value(part_data, train_index, test_index,
                       train_values=None, test_values=None):
    """Partitions a dataset by value
    
    Ensures separation by value of the test and training samples. Each
    input index is a bool index listing the candidate samples. The
    function sets the train index to `False` for samples that do not
    have a valid train value and test index to `False` for samples that
    do not have a valid test value.
    
    Parameters
    ----------
    part_data : pd.Series
        The column to use to partition the data.
    train_index : np.array
        1-dimensional bool array the same length as part_data.
        Samples in the training set should be set to True, all others
        to False.
    test_index : np.array
        1-dimensional bool array the same length as part_data.
        Samples in the test set should be set to True, all others
        to False.
    train_values : list
        The list of values to include in the training data. If ``None``
        (or anything other than a list), the train_index and val_indexes
        are returned unchanged
    test_values : list
        The list of values to include in the test data. Values may
        overlap with train_values. If ``None`` (or anything other than a
        list), the test_index is returned unchanged.

    Returns
    -------
    train_index : np.array
        The updated training index with all samples for the test years
        set to False.
    test_index : np.array
        The updated test index with all samples for years other than
        the test years set to False.

    """
    if test_index is not None:
        test_adjust = part_data.isin(test_values)
        test_index = test_index & test_adjust
    train_adjust = part_data.isin(train_values)
    train_index = train_index & train_adjust
    return train_index, test_index
    
	
def random_split(num_samples, test_size, random_seed=None):
    """Splits samples randomly
    
    Randomly assigns numbers in range(0, num_samples) to the test and
    training sets in accordance with the required split
    sizes. Returns an index for each set that indicates which samples
    are in each set.

    Parameters
    ----------
    num_samples : int
        The number of samples.
    test_size : float or int
        The size of the test set. If int, the number of samples. If
        float, the proportion of samples.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the training set, false otherwise.
    test_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the test set, false otherwise.
    """
    if random_seed is not None:
        random.seed(random_seed)
    sample_list = list(range(num_samples))
    random.shuffle(sample_list)

    # Generate the test index
    if type(test_size) is int:
        break1 = test_size
    else:
        break1 = int(np.floor(num_samples * test_size))
    test_index = np.zeros(num_samples, dtype=bool)
    test_index[sample_list[:break1]] = True

    # Generate the training index
    train_index = np.zeros(num_samples, dtype=bool)
    train_index[sample_list[break1:]] = True
    return train_index, test_index


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
    idx, _ = random_split(train_val_idx.sum(), val_size, random_seed)
    train_index = np.array([False] * num_samples)
    val_index = np.array([False] * num_samples)
    pos = 0
    for i, x in enumerate(train_val_idx):
        if x:
            train_index[i] = idx[pos]
            val_index[i] = not idx[pos]
            pos += 1
    return train_index, val_index


def filter_index(samples, filter_params, match_sources=None, train_index=None, random_seed=None):
    filter_ = np.ones(samples.shape[0])
    if filter_params:
        filter_column = filter_params.get('column')
        filter_method = filter_params.get('method')
        filter_params = filter_params.get('params')
        if filter_method == 'matchTest':
            if match_sources and filter_column:
                # For multi-source input, select samples where the filter_column value matches
                # the range of values present in the match_sources inputs
                filter_ = samples.loc[match_sources][filter_column].drop_duplicate()
                filter_ = samples[filter_column].isin(filter_)
        elif filter_method == 'random':
            if train_index is None:
                # Get a fixed number of samples for training/testing
                _, filter_ = random_split(samples.shape[0], filter_params[0], random_seed)
            else:
                # We want a fixed number of samples from the training set, which is identified by
                # train_index. We get them by doing a train_val_split, which returns the required
                # number of samples in the second position
                _, filter_ = train_val_split(samples.shape[0], filter_params[0],
                                             train_index, random_seed)
        else:
            filter_str = ', '.join([str(x) for x in filter_params])
            print(f"Samples filtered using \"{filter_column}.{filter_method}({filter_str})\"")
            filter_ = getattr(samples[filter_column], filter_method)(*filter_params)
    return filter_


def split_by_column(data, test_size, train_index=None, split_col='Site', stratify=None,
                    min_sites=6, random_seed=None):
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
    test_size : float or int
        The size of the test set. If int, the number of samples. If
        float, the proportion of samples.
    split_col : str, optional
        The name of the column containing the sample site. The default
        is 'Site'.
    stratify : str or None, optional
        The name of the column to use for stratified splitting. None
        for no stratified splits. The default is None.
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
    test_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the test set, false otherwise.
    """
    # Create the groups
    group_data = data if train_index is None else data[train_index]
    if stratify:
        groups =_stratify_split(group_data, split_col, stratify, min_sites)
        y = groups.stratify
        Splitter = StratifiedShuffleSplit
        
    else:
        print(f'Split by {split_col}, no stratify')
        groups = group_data.groupby([split_col], as_index=False).agg(counts=(split_col, 'count'))
        y = groups[split_col]
        Splitter = ShuffleSplit

    # Generate the indexes
    temp = data.reset_index()[split_col]
    if type(test_size) is float:
        test_size = int(np.floor(y.size * test_size))
    sss = Splitter(n_splits=1, test_size=test_size, random_state=random_seed)
    train_values, test_values = next(sss.split(y, y))
    if train_index is None:
        train_index = np.ones(temp.size, dtype=bool)
    test_index = train_index.copy()
    train_index[temp.isin(groups.loc[test_values, split_col]).values] = False
    test_index[temp.isin(groups.loc[train_values, split_col]).values] = False
    return train_index, test_index	


def split_by_source(data, test_sources, train_sources=None, random_seed=None):
    """Splits sample sites randomly.
    
    Splits the data into training and test sets by source.
    
    Parameters
    ----------
    data : DataFrame
        Data Frame of samples. Must have an index or column named
        ``Source``, which will be used to split the data.
    test_sources : list
        A list of sources to use for the test data.
    train_sources : list, optional
        A list of sources to use for the training data. If ``None``, all
        sources that are not test sources are used for training. The
        default is None.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the training set, false otherwise.
    test_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the test set, false otherwise.
    """
    temp = data.reset_index().Source
    
    # Generate the test index
    test_index = np.zeros(temp.size, dtype=bool)
    test_index[temp.isin(test_sources)] = True
    
    # Generate the train index
    if train_sources:
        train_index = np.zeros(temp.size, dtype=bool)
        train_index[temp.isin(train_sources)] = True
    else:
        train_index = np.ones(temp.size, dtype=bool)
        train_index[temp.isin(test_sources)] = False

    return train_index, test_index	


def kfold_indexes(data, n_folds, test_folds=1, split_col='Site', stratify=None,
                  min_values=6, random_seed=None):
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
                test_ = np.union1d(test_, folds[y]['test'])
                train_ = np.setdiff1d(train_, folds[y]['test'])
            new_folds.append({'train': train_, 'test': test_})
        return new_folds
        
        
    if stratify:
        split_values =_stratify_split(data, split_col, stratify, min_values)
        y = split_values.stratify
        Splitter = StratifiedKFold
        
    else:
        print(f'Split by {split_col}, no stratify')
        split_values = data.groupby([split_col], as_index=False).agg(counts=(split_col, 'count'))
        y = split_values[split_col]
        Splitter = KFold

    # Create the folds
    sss = Splitter(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds = [{'train': i1, 'test': i2} for i1, i2 in sss.split(y, y)]
    if test_folds > 1:
        folds = multi_folds(folds, test_folds)
        n_folds = len(folds)
    
    # Generate the indexes
    temp = data.reset_index()[split_col]
    train_index = [np.zeros(temp.size, dtype=bool) for _ in range(n_folds)]
    test_index = [np.zeros(temp.size, dtype=bool) for _ in range(n_folds)]
    for n, fold in enumerate(folds):
        train_index[n][temp.isin(split_values.loc[fold['train'], split_col])] = True
        test_index[n][temp.isin(split_values.loc[fold['test'], split_col])] = True

    return train_index, test_index


def _get_split_indexes(model_params, samples):
    split_type = model_params['splitMethod']
    if split_type == 'bySource':
        if model_params['testSources']:
            train_index, test_index = split_by_source(
                    data=samples,
                    test_sources=model_params['testSources'],
                    train_sources=model_params['trainSources'],
                    random_seed=model_params['randomSeed'])
        else:
            raise ValueError("testSources parameter required with 'bySource' splits")
    else:
        if model_params['testSize'] <= 0:
            raise ValueError(f"testSize parameter must be > 0 for single {split_type} split")
        if split_type == 'random':
            train_index, test_index = random_split(
                    num_samples=samples.shape[0],
                    test_size=model_params['testSize'],
                    random_seed=model_params['randomSeed'])
        elif split_type == 'byValue':
            train_index, test_index = split_by_column(
                    data=samples,
                    test_size=model_params['testSize'],
                    split_col=model_params['splitColumn'],
                    stratify=model_params['splitStratify'],
                    random_seed=model_params['randomSeed'])
        else:
            raise ValueError(f"Invalid train/test split method: {split_type}")
    return train_index, test_index


def split_data(data, train_index, test_index, val_index=None, **unused):
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
    train_data = data[train_index] if train_index is not None else None
    val_data = data[val_index] if val_index is not None else None
    test_data = data[test_index] if test_index is not None else None
    return train_data, test_data, val_data


def _get_data_by_source(X, y, train_index, val_index, test_index,
                        sample_sources, train_sources, test_sources):
    data = {'train': {}, 'val':   {}, 'test':  {}}

    test_index = None if test_index is None else (
        test_index & sample_sources.isin(test_sources).values)
    _, y_test, _ = split_data(y, None, test_index, None)
    data['test'] = {'X': {}, 'y': y_test}
    for input_, xdata in X.items():
        _, test, _ = split_data(xdata, None, test_index, None)
        data['test']['X'][input_] = test

    for source in train_sources:
        trn_ind = train_index & sample_sources.eq(source).values
        val_ind = None if val_index is None else (
            val_index & sample_sources.eq(source).values)
        y_train, _, y_val = split_data(y, trn_ind, None, val_ind)
        data['train'][source] = {'X': {}, 'y': y_train}
        data['val'][source] = {'X': {}, 'y': y_val}
        for input_, xdata in X.items():
            train, _, val = split_data(xdata, trn_ind, None, val_ind)
            data['train'][source]['X'][input_] = train
            data['val'][source]['X'][input_] = val
    return data


def _get_data(model_params, samples, X, y, parent_results,
              train_index, val_index, test_index, fold=''):
    if parent_results is not None:
        parent_filter = model_params['inputs']['parent'].get('filter')
        try:
            parent_data = parent_results.base
            samples_index = _samples_filter(parent_filter, samples, X, parent_data) #, parent_normal)
        except:
            year = fold.split('_')[0]
            parent_data = parent_results[f'{year}_base']
            samples_index = _samples_filter(parent_filter, samples, X, parent_data) #, parent_normal)
        train_index = train_index & samples_index
        test_index = None if test_index is None else (test_index & samples_index)
        val_index = None if val_index is None else (val_index & samples_index)

    if isinstance(model_params['epochs'], int):
        y_train, y_test, y_val = split_data(y, train_index, test_index, val_index)
        data = {'train': {'X': {}, 'y': y_train},
                'val':   {'X': {}, 'y': y_val},
                'test':  {'X': {}, 'y': y_test}}
        for source, xdata in X.items():
            train, test, val = split_data(xdata, train_index, test_index, val_index)
            data['train']['X'][source] = train
            data['val']['X'][source] = val
            data['test']['X'][source] = test
    else:
        train_sources = list(model_params['epochs'].keys())
        test_sources = model_params['testSources']
        data = _get_data_by_source(X, y, train_index, val_index, test_index,
                                   samples.reset_index().Source, train_sources, test_sources)
    return data


def train_test_split(model_params, samples, X, y, parent_results):
    """Splits data into test and training sets

    Splits data into test, training and optionally validation sets. The
    model parameters determine how to split the data.
    
    Note: this function makes a single train/test split. kfold_split
    is used if multiple train/test splits are required.
    
    valSize:
        If > 0, indicates if a validation set should be created.
    
    splitMethod:
      - random: samples are split randomly
      - byValue: sites are split into test and training sites, then all
        samples from the site are allocated to the respective set
      - byYear: samples are allocated to the test and training sets
        based on the sampling year
    
    valSize:
        The proportion or number of samples to allocate to the validation
        set. Used for random and byValue splits.
    
    splitColumn:
        The column in the samples containing the site. Used for byValue
        splits.
    
    yearColumn:
        The column in the samples containing the sampling year. Used
        for byYear splits.
    
    splitStratify:
        Indicates the column to use for stratified splitting, if this
        is needed. Stratified splitting groups the sites by value and
        ensures each split contains a roughly equal proportion of each
        group. If None, stratified splittig is not used. Used for
        byValue splits.
    
    splitYear:
      - If splitMethod is "byYear" and splitFolds is 0: Samples for
        this year or later go into the test set. Samples earlier than
        this year go into the training set.
      - If splitMethod is "byYear" and splitFolds is 1: Samples for
        this year go into the test set. Samples earlier than this year
        go into the training set. Samples for later than this year are
        not used.
      - If splitMethod is "random" or "byValue": Any samples in the test
        set for earlier than this year are removed. Any samples in the
        training or validation sets for this year or later are removed.
                
    randomSeed:
        The seed for the random number generator. Used for random and
        byValue splits.
    
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
    def get_yearly_indexes(model_params, samples):
        full_index = pd.Series([True] * samples.shape[0], index=samples.index)
        train_index, test_index = partition_by_year(
            samples[model_params['yearColumn']],
            model_params['splitYear'],
            full_index, full_index,
            num_years=model_params['splitFolds'],
            test_all_years=model_params['testAllYears'],
            train_adjust=model_params['trainAdjust'],
            test_adjust=model_params['testAdjust'])
        return train_index, test_index

    split_type = model_params['splitMethod']
    if split_type in ['random', 'byValue', 'bySource']:
        train_index, test_index = _get_split_indexes(model_params, samples)
        sample_values = samples[model_params['splitColumn']] if model_params['splitMax'] else None
        if model_params['splitYear']:
            train_index, test_index = partition_by_year(
                samples[model_params['yearColumn']],
                model_params['splitYear'],
                train_index, test_index,
                sample_values = sample_values,
                test_all_years=model_params['testAllYears'],
                train_adjust=model_params['trainAdjust'],
                test_adjust=model_params['testAdjust'])
    elif split_type in ['byYear', None]:
        if model_params['splitYear']:
            train_index, test_index = get_yearly_indexes(model_params, samples)
        else:
            test_index = None
            train_index = np.array([True] * samples.shape[0])
    else:
        raise ValueError(f"Invalid train/test split method: {split_type}")

    if (isinstance(model_params['samplesFilter'], dict)
        and model_params['samplesFilter']['apply'].lower().startswith('train')):
        filter_ = filter_index(samples, model_params['samplesFilter'],
                               match_sources=model_params['testSources'],
                               train_index=train_index,
                               random_seed=model_params['randomSeed'])
        train_index = train_index & filter_
    
    if model_params['valSize'] > 0:
        if split_type == 'byValue':
            train_index, val_index = split_by_column(
                    data=samples,
                    test_size=model_params['valSize'],
                    train_index=train_index,
                    split_col=model_params['splitColumn'],
                    stratify=model_params['splitStratify'],
                    random_seed=model_params['randomSeed'])
        else:
            train_index, val_index = train_val_split(
                    num_samples=samples.shape[0],
                    val_size=model_params['valSize'],
                    train_val_idx=train_index,
                    random_seed=model_params['randomSeed'])
    else:
        val_index = None

    data = _get_data(model_params, samples, X, y, parent_results,
                     train_index, val_index, test_index)
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

    valSize:
        If > 0, indicates if a validation set should be created.
    
    splitMethod:
      - random: samples are split randomly
      - byValue: sites are split into test and training sites, then all
        samples from the site are allocated to the respective set
      - NOTE: byYear cannot be used with k-fold splits
    
    splitFolds:
        The number of folds (``K``) required.
        
    valSize:
        The proportion or number of samples to allocate to validation
        sets.
    
    splitColumn:
        The column in the samples containing the site. Used for byValue
        splits.
    
    splitStratify:
        Indicates the column to use for stratified splitting, if this
        is needed. Stratified splitting groups the sites by value and
        ensures each split contains a roughly equal proportion of each
        group. If None, stratified splittig is not used. Used for
        byValue splits.
    
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
        if model_params['splitMethod'] == 'random':
            train_index, test_index = kfold_indexes(
                    data=samples.reset_index(),
                    n_folds=model_params['splitFolds'],
                    test_folds=model_params['testFolds'],
                    split_col=samples.index.name or 'index',
                    stratify=False,
                    random_seed=model_params['randomSeed'])
            folds = range(len(train_index))
        elif model_params['splitMethod'] == 'byValue':
            train_index, test_index = kfold_indexes(
                    data=samples,
                    n_folds=model_params['splitFolds'],
                    test_folds=model_params['testFolds'],
                    split_col=model_params['splitColumn'],
                    stratify=model_params['splitStratify'],
                    random_seed=model_params['randomSeed'])
            folds = range(len(train_index))
        elif model_params['splitMethod'] == 'bySource' and model_params['testSources']:
            train_index = {}
            test_index = {}
            folds = model_params['sourceNames']
            for source in model_params['sourceNames']:
                train_index[source], test_index[source] = split_by_source(
                        data=samples,
                        test_sources=[source],
                        train_sources=None,
                        random_seed=model_params['randomSeed'])
        else:
            raise ValueError(
                f"testSources ({model_params['testSources']}) required with 'bySource' splits")

        return folds, train_index, test_index

    def get_yearfold_indexes(model_params, samples, folds, train_index, test_index):
        sample_values = samples[model_params['splitColumn']] if model_params['splitMax'] else None
        year_folds = model_params.get('yearFolds', 0)
        if year_folds <= 1:
            for n in folds:
                train_index[n], test_index[n] = partition_by_year(
                    samples[model_params['yearColumn']],
                    model_params['splitYear'],
                    train_index[n], test_index[n],
                    num_years=year_folds,
                    sample_values=sample_values,
                    test_all_years=model_params['testAllYears'],
                    train_adjust=model_params['trainAdjust'],
                    test_adjust=model_params['testAdjust'])
        else:
            first_year = model_params['splitYear']
            last_year = first_year + year_folds
            year_folds = list(reversed(range(first_year, last_year)))
            trn_ind = {}
            tst_ind = {}
            new_folds = []
            for n in folds:
                for year in year_folds:
                    if len(folds) > 1:
                        fold = f'{year}_{n:02d}' if isinstance(n, int) else f'{year}_{n}'
                    else:
                        fold = str(year)
                    new_folds.append(fold)
                    trn_ind[fold], tst_ind[fold] = partition_by_year(
                        samples[model_params['yearColumn']],
                        year,
                        train_index[n], test_index[n],
                        num_years=1,
                        sample_values=sample_values,
                        test_all_years=model_params['testAllYears'],
                        train_adjust=model_params['trainAdjust'],
                        test_adjust=model_params['testAdjust'])
            folds = new_folds
            train_index = trn_ind
            test_index = tst_ind

        return folds, train_index, test_index
    
    def get_yearly_indexes(model_params, samples):
        train_index = {}
        test_index = {}
        first_year = model_params['splitYear']
        last_year = first_year + model_params['yearFolds']
        folds = list(reversed(range(first_year, last_year)))
        for year in folds:
            full_index = pd.Series([True] * samples.shape[0], index=samples.index)
            train_index[year], test_index[year] = partition_by_year(
                samples[model_params['yearColumn']],
                year, full_index, full_index,
                num_years=1,
                test_all_years=model_params['testAllYears'],
                train_adjust=model_params['trainAdjust'],
                test_adjust=model_params['testAdjust'])
        return folds, train_index, test_index

    def load_folds(samples, fold_params, val_data=None):
        # Assumes only one source is used
        fold_params = fold_params if isinstance(fold_params, (list, tuple)) else [fold_params]
        model_dir = fold_params[0]
        source = fold_params[1] if len(fold_params) > 1 else None

        all_folds = {}
        all_data = {}
        all_indexes = {}
        for ds in ['test', 'train', 'val']:
            with open(os.path.join(model_dir, f'{ds}_folds.json')) as f:
                all_folds[ds] = json.load(f)
            all_data[ds] = []
            for fold_name, fold_data in all_folds[ds].items():
                zip_ = zip([fold_name]*len(fold_data), fold_data)
                if fold_data and isinstance(fold_data[0], list):
                    all_data[ds].extend([[x, y[0], y[1]] for (x, y) in zip_])
                else:
                    all_data[ds].extend([[x, source, y] for (x, y) in zip_])
            all_data[ds] = pd.DataFrame(all_data[ds], columns=['Fold', 'Source', 'ID'])
            all_data[ds] = all_data[ds].set_index(['Fold', 'Source']).sort_index()
            all_indexes[ds] = {}
            for fold in all_folds[ds].keys():
                try:
                    if ds == 'val' and not val_data:
                        all_indexes[ds][fold] = None
                    else:
                        all_indexes[ds][fold] = samples.index.isin(
                            all_data[ds].loc[fold, source].ID)
                except:
                    all_indexes[ds][fold] = np.array([False] * samples.shape[0])
        folds = list(all_folds['train'].keys())

        if (isinstance(model_params['samplesFilter'], dict)
            and model_params['samplesFilter'].get('apply', 'all').lower().startswith('train')):
            for fold in folds:
                filter_ = filter_index(samples, model_params['samplesFilter'],
                                       match_sources=model_params['testSources'],
                                       train_index=all_indexes['train'][fold],
                                       random_seed=model_params['randomSeed'])
                all_indexes['train'][fold] = all_indexes['train'][fold] & filter_
        return folds, all_indexes['train'], all_indexes['val'], all_indexes['test']
    
    def generate_folds(model_params, samples):
        if model_params['splitMethod'] in ['random', 'byValue', 'bySource']:
            if model_params['splitFolds'] > 1:
                folds, train_index, test_index = get_fold_indexes(model_params, samples)
            else:
                train_index, test_index = [
                    [idx] for idx in _get_split_indexes(model_params, samples)]
                folds = range(1)

            if model_params['testSources']:
                temp = samples.reset_index(level='Source').Source
                for fold in folds:
                    test_index[fold][~temp.isin(model_params['testSources'])] = False

            if model_params['splitYear']:
                folds, train_index, test_index = get_yearfold_indexes(
                    model_params, samples, folds, train_index, test_index)

        elif model_params['splitMethod'] in ['byYear', None]:
            model_params['yearFolds'] = model_params.get('yearFolds', 0) \
                                        or model_params['splitFolds']
            folds, train_index, test_index = get_yearly_indexes(model_params, samples)
        else:
            raise ValueError(f"Invalid train/test split method: {model_params['splitMethod']}")
            
        if (isinstance(model_params['samplesFilter'], dict)
            and model_params['samplesFilter'].get('apply', 'all').lower().startswith('train')):
            for fold in folds:
                filter_ = filter_index(samples, model_params['samplesFilter'],
                                       train_index=train_index[fold],
                                       match_sources=model_params['testSources'],
                                       random_seed=model_params['randomSeed'])
                train_index[fold] = train_index[fold] & filter_

        if isinstance(train_index, list):
            val_index = [None for fold in folds]
        else:
            val_index = {fold: None for fold in folds}

        if model_params['valSize'] > 0:
            for fold in folds:
                if model_params['splitMethod'] == 'byValue':
                    train_index[fold], val_index[fold] = split_by_column(
                            data=samples,
                            test_size=model_params['valSize'],
                            train_index=train_index[fold],
                            split_col=model_params['splitColumn'],
                            stratify=model_params['splitStratify'],
                            random_seed=model_params['randomSeed'])
                else:
                    train_index[fold], val_index[fold] = train_val_split(
                            num_samples=samples.shape[0],
                            val_size=model_params['valSize'],
                            train_val_idx=train_index[fold],
                        random_seed=model_params['randomSeed'])
        return folds, train_index, val_index, test_index
        
    def save_folds(model_params, samples, folds, train_index, val_index, test_index):
        def save_foldset(folds, fold_index, save_dir, save_file):
            with open(os.path.join(save_dir, save_file), 'w') as f:
                save_index = fold_index.copy()
                for fold in folds:
                    if save_index[fold] is None:
                        save_index[fold] = []
                    else:
                        save_index[fold] = samples.index[save_index[fold]].tolist()
                json.dump(save_index, f, indent=2)

        kfold_split.folds = folds
        kfold_split.indexes = {'train': train_index, 'val': val_index, 'test': test_index}
        if model_params.get('run', None) is None or model_params['resplit']:
            save_dir = model_params['modelDir']
        else:
            save_dir = os.path.dirname(model_params['modelDir'].rstrip('\\/'))
        save_foldset(folds, test_index, save_dir, 'test_folds.json')
        if model_params['saveFolds'] or model_params['saveTrain']:
            save_foldset(folds, train_index, save_dir, 'train_folds.json')
        if model_params['saveFolds'] or model_params['saveValidation']:
            save_foldset(folds, val_index, save_dir, 'val_folds.json')
    
    if not kfold_split.folds or model_params['resplit']:
        get_folds = model_params.get('loadFolds')
        if get_folds:
            folds, train_index, val_index, test_index = load_folds(
                samples, get_folds, model_params['valSize'])
        else:
            folds, train_index, val_index, test_index = generate_folds(model_params, samples)

        save_folds(model_params, samples, folds, train_index, val_index, test_index)
        if isinstance(model_params['sourceNames'], list):
            kfold_split.source = samples.reset_index().Source
        if model_params['modelRuns'] < 0:
            yield None

    else:
        folds = kfold_split.folds
        train_index = kfold_split.indexes['train']
        val_index = kfold_split.indexes['val']
        test_index = kfold_split.indexes['test']

    all_folds = model_params['saveModels'] or model_params['saveTrain'] \
                or model_params['saveValidation']
    for fold in folds:
        data = _get_data(model_params, samples, X, y, parent_results,
                         train_index[fold], val_index[fold], test_index[fold], fold)
        ytest = data['test']['y']
        if all_folds or not ((ytest is None) or (ytest.shape[0] == 0)):
            yield (fold, data)
            
# Add attributes to kfold_split to store the fold indexes between runs
kfold_split.indexes = None
kfold_split.folds = None
