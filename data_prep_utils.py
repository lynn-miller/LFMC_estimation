"""Model building and evaluation utilities"""

import json
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

   
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


def percentiles_to_bounds(data, percentiles):
    """Gets the percentile bounds from an array
    
    Gets the percentile bounds for a 2-d or 3-d array. Dimensions are
    assumed to be [instances, time-steps, features], so for non-TS data
    of size n*m, reshape to (n, 1, m).

    Parameters
    ----------
    data : np.array
        Input dataset. Must have at least two dimensions.
    percentiles : int, tuple-like or list-like
        Percentiles for bounds. If int, this is assumed to be the lower
        percentile, and the upper percentile is 100 - lower percentile.
        If tuple-like, it should be the lower and upper percentiles. If
        list-like, should be a list in the format [`lower`, `upper`]
        where `lower` and `upper` are arrays of length `m` and specify
        the lower and upper percentiles for each feature. `Upper` is 
        optional and is set to `100 - lower` if not provided.

    Returns
    -------
    np.array
        The bounds, calculated across the first two dimensions.

    """
    if (type(percentiles) is int) or (len(percentiles) == 1):
        percentiles = [percentiles, 100-percentiles]
    if type(percentiles[0]) is int:
        if data.ndim == 1:
            bounds = np.percentile(data, percentiles)
        else:
            bounds = np.percentile(data, percentiles, axis=(0, 1))
    else:
        lower = [np.percentile(data[:,:,n], p, axis=(0,1))
                 for n, p in enumerate(percentiles[0])]
        upper = [np.percentile(data[:,:,n], p, axis=(0,1))
                 for n, p in enumerate(percentiles[1])]
        bounds = np.array([lower, upper])
    return bounds


def save_bounds(bounds, input_name='', model_dir=''):
    """Saves the normalisation bounds.
    
    Save the normalisation bounds in CSV format to file
    ``<model_dir>/<input_name>_bounds.csv``
    
    Parameters
    ----------
    bounds : np.ndarray
        Two-dimensional array. First dimension is the lower and upper
        bounds, second dimension is the bounded channels or features.
    input_name : str
        Name of the input. Used for deriving the save file name.
    model_dir : str
        Name of the directory where the bounds are to be saved.

    Returns
    -------
    None.

    """
    bounds_file = os.path.join(model_dir, f'{input_name}_bounds.csv')
    np.savetxt(bounds_file, bounds, delimiter=',')


def load_bounds(input_name='', model_dir=''):
    """Loads the normalisation bounds.
    
    Load the normalisation bounds from CSV file
    ``<model_dir>/<input_name>_bounds.csv``
    
    Parameters
    ----------
    input_name : str
        Name of the input. Used for deriving the save file name.
    model_dir : str
        Name of the directory where the bounds are saved.

    Returns
    -------
    bounds : np.ndarray
        Two-dimensional array. First dimension is the lower and upper
        bounds, second dimension is the bounded channels or features.

    """
    bounds_file = os.path.join(model_dir, f'{input_name}_bounds.csv')
    return np.genfromtxt(bounds_file, delimiter=',')
    
    
def normalise(data, method='none', percentiles=0, data_range=[0, 10000], scaled_range=(0, 1),
              mean=None, sd=None, clip=False, input_name=None, model_dir=None,
              save_params=False, load_params=False, return_params=False):
    """Normalises the data.
    
    Normalises and optionally clips an array. When using the minMax or
    standard methods, the data must be a 2-d or 3-d array (unused
    dimensions can be size 1) and data is normalised across the first 2
    dimensions. If clipping is requested, data is clipped before being
    normalised.
    
    Parameters
    ----------
    data : np.ndarray
        Array of the data to be normalised.
    method : str, optional
        Normalisation method to use. Currently implemented methods are:
          - minMax: Normalise so the lower percentile is 0 and the
            upper percentile is 1.
          - range: Normal across a set range, so lower range is 0 and
            upper range is 1.
          - standard: Normalise to mean=0, std_dev=1. If `mean` and `sd`
            parameters are provided, assume these values for input data
            mean and std_dev, else calculate these from the data.
          - FW_ratio: Convert the LFMC values to the fresh weight ratio
            (instead of dry weight ratio) and scale to range [-1, 1].
          - none: Default. No normalisation performed. Data is clipped
            if required, otherwise returned unchanged.  
    percentiles : int or list-like, optional
        The percentile to use with the minMax method, If a single int
        value, this is treated as the lower percentile and the upper
        percentile is set to 100 - the value. Arrays of lower and upper
        percentiles can also be provided. See `percentiles_to_bounds`
        for details. The default is 0.
    data_range : tuple or list-like, optional
        Two values corresponding to the lower and upper bounds of the
        normalisation range. The values can be numbers or numpy arrays
        that can be broadcast to `data`. The default is [0, 10000].
    scaled_range : tuple or list-like, optional
        Two values corresponding to the lower and upper bounds of the
        output range. The values can be numbers or numpy arrays that
        can be broadcast to `data`. After normalising, the data is
        re-scaled to this range. The default is (0, 1).
    clip : bool, optional
        Indicates if the data should be clipped to ``range`` (``True``)
        or not (``False``). The default is False.
    input_name : str
        The name of the model input being processed. Required if either
        ``save_bounds`` or ``load_bounds`` are True. The default is
        None.
    model_dir : str
        The name of the model directory. Required if either
        ``save_bounds`` or ``load_bounds`` are True. The default is
        None.
    save_params : bool
        If ``True`` the bounds or mean/sd are saved. The ``input_name``
        and ``model_dir`` parameters must also be provided. The default
        is False.
    load_params : bool
        If ``True``: If the method is ``minMax``, the bounds are loaded
        from saved bounds, rather than computed from the data. If the
        method is ``standard`` the mean and sd are loaded from the
        saved bounds. The ``input_name`` and ``model_dir`` parameters
        must also be provided. The default is False.
    return_params : bool
        If ``True``, a dictionary of parameters is returned. This
        dictionary can be used to update the parameters used on this
        call to ``normalise`` to values that allow another set of data
        to be normalised or denormalised consistently with the current
        dataset. The default is False.
    mean : number or list-like, optional
        For the ``standard`` method. The mean of the denormalised data.
    sd : number or list-like, optional
        For the ``standard`` method. The standard deviation of the
        denormalised data.

    Returns
    -------
    np.ndarray
        The normalised data.
    """
    if clip:
        data = np.clip(data, data_range[0], data_range[1])
    if method == 'minMax':
        if load_params:
            bounds = load_bounds(input_name, model_dir)
        else:
            bounds = percentiles_to_bounds(data, percentiles)
        temp = (data - bounds[0]) / (bounds[1] - bounds[0])
        if input_name and model_dir and save_params:
            save_bounds(bounds, input_name, model_dir)
        params = {'method': 'range', 'data_range': bounds.tolist()}
    elif method == 'range':
        data_range = np.array(data_range)
        temp = (data - data_range[0]) / (data_range[1] - data_range[0])
        params = {}
    elif method == 'standard':
        if data.ndim == 1:
            axes = None
        else:
            axes = tuple(np.arange(data.ndim - 1))
        if load_params:
            mean, sd = load_bounds(input_name, model_dir)
        else:
            if mean is None:
                mean = np.mean(data, axis=axes)
            if sd is None:
                sd = np.std(data, axis=axes)
        if input_name and model_dir and save_params:
            save_bounds(np.array([mean, sd]), input_name, model_dir)
        temp = (data - mean) / sd
        params = {'mean': mean, 'sd': sd}
    elif method == 'FW_ratio':
        temp = 2 * data / (100 + data) - 1
        params = {}
    elif (method == 'none') or (method is None):
        return (data, {}) if return_params else data
    else: # method not implemented
        raise ValueError(f'Invalid method: {method}')
    if tuple(scaled_range) != (0, 1):
        temp = temp * (scaled_range[1] - scaled_range[0]) + scaled_range[0]
    if return_params:
        return temp, params
    else:
        return temp
    
    
def denormalise(data, method='none', data_range=(0, 10000), scaled_range=(0, 1), mean=0, sd=1):
    """De-normalises the data.
    
    Converts normalised data back to de-normalised values. When using
    the minMax or standard methods, the data must be a 2-d or 3-d array
    (unused dimensions can be size 1) and data is de-normalised across
    the first 2 dimensions.
    
    Parameters
    ----------
    data : np.ndarray
        Array of the data to be normalised.
    method : str, optional
        Normalisation method to use. Currently implemented methods are:
          - minMax: Normalise so the lower percentile is 0 and the
            upper percentile is 1.
          - range: Normal across a set range, so lower range is 0 and
            upper range is 1.
          - FW_ratio: Convert the LFMC values from fresh weight ratios
            back to dry weight ratios.
          - none: Default. No normalisation performed. Data is clipped
            if required, otherwise returned unchanged.  
    in_range : tuple or list-like, optional
        Two values corresponding to the lower and upper bounds of the
        normalisation range. The values can be numbers or numpy arrays
        that can be broadcast to `data`. The default is [0, 10000].
    out_range : tuple or list-like, optional
        Two values corresponding to the lower and upper bounds of the
        output range. The values can be numbers or numpy arrays that
        can be broadcast to `data`. After normalising, the data is
        re-scaled to this range. The default is (0, 1).
    mean : number or list-like, optional
        For the ``standard`` method. The mean of the denormalised data.
    sd : number or list-like, optional
        For the ``standard`` method. The standard deviation of the
        denormalised data.

    Returns
    -------
    np.ndarray
        The de-normalised data.
    """
    if (method == 'none') or (method is None):
        return data
    elif method in ['range', 'minMax']:
        if tuple(scaled_range) == (0, 1):
            temp = data
        else:
            temp = (data - scaled_range[0]) / (scaled_range[1] - scaled_range[0])
        temp = temp * (data_range[1] - data_range[0]) + data_range[0]
    elif method == 'standard':
        temp = (data * sd ) + mean
    elif method == 'FW_ratio':
        temp = (1 + data) / (1 - data) * 100
    else: # method not implemented
        raise ValueError(f'Invalid method: {method}')
    return temp


def create_onehot_enc(onehot_cols, fit_data=None, model_dir=None, source_dir=None, save=False):
    """Creates the one-hot encoder
    
    Creates, fits and optionally saves a one-hot encoder. If
    ``fit_data`` is specified, this is used to fit the one-hot encoder.
    If ``fit_data`` is not specified, the ``onehot_data.json`` file in
    ``model_dir`` is loaded and used to create the one-hot encoding
    categories. If ``save`` is ``True``, ``fit_data`` is a DataFrame
    and ``model_dir`` is specified, ``save_onehot_enc`` is called to
    save the data required to re-create the one-hot encoder.

    Parameters
    ----------
    onehot_cols : list
        List of columns to one-hot encode.
    fit_data : pd.DataFrame or np.ndarray, optional
        The data to use to fit the encoder. The default is None.
    model_dir : str, optional
        If ``fit_data`` is none, the directory containing the saved
        one-hot encoder data. If ``save`` is ``True``, the directory
        where this data should be saved. The default is None.
    save : bool, optional
        Whether or not to save the one-hot encoder data. The default is
        False.

    Returns
    -------
    onehot_enc : sklearn.OneHotEncoder
        The fitted one-hot encoder.

    """
    if fit_data is None:
        source_dir = source_dir or model_dir
        file_name = os.path.join(source_dir, 'onehot_data.json')
        with open(file_name, 'r') as f:
            onehot_dict = json.load(f)
        fit_data = [[onehot_dict[cat][0] for cat in onehot_cols]]   # dummy data to fit the encoder
    else:
        fit_df = pd.DataFrame(fit_data, columns=onehot_cols)
        onehot_dict = {c[0]: c[1].dropna().unique().tolist() for c in fit_df.items()}

    if save and ((fit_data is not None) or (source_dir != model_dir)):
        file_name = os.path.join(model_dir, 'onehot_data.json')
        with open(file_name, 'w') as f:
            json.dump(onehot_dict, f, indent=2)

    # Set categories to a nested list of the values to encode for each variable. Then
    # create dummy data for fitting the encoder. The encoder will use the categories
    # to create the encoding, rather than the fitting data.
    categories = []
    for col_name in onehot_cols:
        categories.append(onehot_dict[col_name])  # column values for encoder
    onehot_enc = OneHotEncoder(categories=categories, handle_unknown='ignore',
                               sparse=False, dtype='int')
    onehot_enc.fit(np.array(fit_data))
    return onehot_enc


def ordinal_encoder(data):
    enc = OrdinalEncoder()
    enc.fit(data)
    return enc.transform(data)


def set_thresholds(samples, thresholds):
    th = thresholds.copy()
    default = th.pop('default')
    threshold_column = (set(th.keys()) - set(['Thresholds'])).pop()
    thresholds_df = pd.DataFrame(th).explode(threshold_column, True)
    index_name = samples.index.name or 'index'
    samples2 = samples.reset_index().merge(thresholds_df, 'left').set_index(index_name)
    return samples2['Thresholds'].fillna(default)


def reweight_source(labels, source, target):
    bin_size = 1
    weights = pd.Series([1.0] * labels.shape[0], index=labels.index)
    min_ = 0 
    max_ = int(labels.max()) + 1
    source_count =labels[source].count()
    target_count = labels[target].count()
    bin_t = 0
    for b in range(min_, max_, bin_size):
        bin_s = labels[source].between(b, b+bin_size, inclusive='left').sum()
        bin_t += labels[target].between(b, b+bin_size, inclusive='left').sum()
        if bin_s == 0:
            continue
        else:
            weight = (bin_t / target_count) / (bin_s / source_count)
            bin_t = 0
        weights[source][labels[source].between(b, b+bin_size, inclusive='left')] = weight
    return weights