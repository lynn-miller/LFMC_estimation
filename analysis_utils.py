"""Result analysis utilities."""
import numpy as np
import os
import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt

from scipy import stats
from scipy.interpolate import interpn

from sklearn.metrics import r2_score, accuracy_score, average_precision_score
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, jaccard_score


def plot_results(fig_name, y, yhat, top_text=None, bottom_text=None,
                 lower=0, upper=300, interpolate=False, bins=30,
                 diagonal=False, regress=False, regress_eqn=False, show_grid=False,
                 ax=None, cmap=None, vmin=None, vmax=None):
    """Create a plot of labels versus predictions
    
    Create a plot of labels versus predictions. The figure (or axes) is
    not displayed, but returned so the caller can display or save it as
    required.
    
    A 2-D histogram of the labels and predictions is generated, and
    optionally interpolated to give a smoothed result. The plot is
    coloured according to the number of (interpolated) samples at each
    

    Parameters
    ----------
    fig_name : str
        A name for the figure. Used to generate a title and filename
        for the figure.
    y : array
        The sample labels.
    yhat : array
        The sample predictions.
    top_text : str or dict, optional
        If str: The text to display in the top-left corner of the plot.
        If dict: The evaluation statistics. The text displayed is
        derived from these metrics.
        If None: No text is displayed. The default is None.
    bottom_text : str, optional
        If str: The text to display in the bottom-right plot corner.
        If None: No text is displayed. The default is None.
    lower : int, optional
        The lower bound to use for binning. The default is 0.
    upper : int, optional
        The upper bound to use for binning. The default is 300.
    interpolate : bool, optional
        True: Interpolate the result before plotting to smooth the plot
        points.
        False: Plot the raw results. The default is False.
    bins : int, optional
        The number of bins (in each dimension) to use when binning the
        data. Ignored if interpolate is False. The default is 30.
    diagonal : bool, optional.
    regress : bool or str, optional.
        Plot the regression line, if truthy. If type str, then assumed
        to be the plot colour. If True, the default colour is used.
    show_grid : bool, optional
    ax : MatPlotLib axes, optional
        An axes for the figure. Used if the plot is to be added as a
        sub-plot to a figure. If this is set, a figure object is not
        created and an axes object is returned. The default is None.
    cmap : Colormap, optional
    vmin : int, optional
        The minimum bin size. Only used for the colour scale, so does
        not cause an error if the number of values in a bin is less
        than this limit. The default is 0.
    vmax : int, optional
        The maximum bin size. Only used for the colour scale, so does
        not cause an error if the number of values in a bin exceeds
        this limit. The default is 25.

    Returns
    -------
    MatPlotLib axes or figure
        Returns the axes if one specified, else the figure.

    """
    FIG_SIZE = 5
    FONT_SIZE = 7
    TITLE_SIZE = 8
    TEXT_OFFSET = 5
    REGRESS_OFFSET = 25
    MARKER_SIZE = 2
    
    def interpolate_data(plot_data, upper, lower, bins):
        plot_data = plot_data.copy()
        bin_width = (upper-lower)//bins
        data, x_bins, y_bins = np.histogram2d(plot_data.y,
                                              plot_data.yhat,
                                              bins=list(range(lower, upper+1, bin_width)),
                                              density=False)
        z = interpn((x_bins[:-1]+(bin_width/2), y_bins[:-1]+(bin_width/2)),
                    data/(bin_width**2),
                    plot_data.to_numpy(),
                    method = "splinef2d",
                    bounds_error=False)
        z[np.where(np.isnan(z))] = 0.0
        plot_data['z'] = z
        return plot_data.drop_duplicates().sort_values('z')

    def add_text_boxes(ax, tl, br, lower, upper):
        if isinstance(tl, (dict, pd.Series)):
            tl = f"$RMSE: {tl['RMSE']:.2f}\%$\n" \
                 f"$Bias: {tl['Bias']:.2f}\%$\n" \
                 f"$R^2: {tl['R2']:.2f}$"
        if tl is not None:
            xloc = lower + TEXT_OFFSET
            yloc = upper - TEXT_OFFSET
            ax.text(xloc, yloc, tl, size=FONT_SIZE, va="top", ha="left")
        if br is not None:
            yloc = lower + TEXT_OFFSET
            xloc = upper - TEXT_OFFSET
            ax.text(xloc, yloc, br, size=FONT_SIZE, va="bottom", ha="right")

    def add_regression_line(ax, y, yhat, lower, upper, colour, eqn):
        if colour is True:
            colour = 'firebrick'
        slope, intercept, _, _, _ = stats.linregress(y, yhat)
        linex = np.asarray(list(range(lower + REGRESS_OFFSET, upper - REGRESS_OFFSET)))
        ax.plot(linex, linex * slope + intercept, '--', color=colour)
        if eqn:
            text = f"y={slope:.2f}x+{intercept:.2f}"
            xloc = (upper - lower) // 2
            yloc = lower + 50
            ax.text(xloc, yloc, text, size=FONT_SIZE, color=colour)

    def label_plot(ax, fig_name, lower, upper):
        ax.axis([lower, upper, lower, upper])
        ax.set_ylabel('Estimated LFMC (%)', fontsize=FONT_SIZE)
        ax.set_xlabel('Measured LFMC (%)', fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.set_title(fig_name, fontsize=TITLE_SIZE)

    # Create the sub-plot, if none
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    else:
        fig = None

    # Plot the data
    ax.grid(show_grid, zorder=0)
    ax.set_axisbelow(True)
    plot_data = pd.DataFrame({'y': np.round(y), 'yhat':np.round(yhat)})
    plot_data = plot_data[(plot_data.y.between(lower, upper))
                          & (plot_data.yhat.between(lower, upper)) ]
    if interpolate:
        plot_data = interpolate_data(plot_data, upper, lower, bins)
    else:
        plot_data['z'] = 0
        plot_data = plot_data.groupby(['y', 'yhat'], as_index=False).count().sort_values('z')
    ax.scatter(plot_data.y, plot_data.yhat, c=plot_data.z, s=MARKER_SIZE,
               cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot_data = plot_data

    # Add other plot components
    if diagonal:
        ax.plot([lower, upper], [lower, upper], ':', color=(0.5, 0.5, 0.5))
    ax.grid(show_grid, zorder=10)
    add_text_boxes(ax, top_text, bottom_text, lower, upper)
    if regress:
        add_regression_line(ax, y, yhat, lower, upper, regress, regress_eqn)
    label_plot(ax, fig_name, lower, upper)

    # Return the axes or figure
    if fig is None:
        return ax
    else:
        return fig


def calc_statistics(y, yhat, classify=False, metrics=None, ybar=None, threshold=0.5, precision=2):
    """Calculate model evaluation statistics
    
    These metrics are available for regression results:
      - Bias: The difference between the mean prediction and the mean
        label
      - RMSE: The root mean squared error of the predictions
      - RMSPE: the root mean squared proportionate error of the predictions
      - ubRMSE (Unbiased RMSE): The RMSE obtained if each prediction is
        adjusted by the Bias
      - R: The correlation coefficient
      - R2 (R-squared): The percent of variance explained

    These metrics are available for classification results:
        'Acc', 'F1Score', 'Kappa', 'AvgPrecision', 'LogLoss', 'AuROC'
      - Acc: Accuracy of the predictions
      - F1Score: The F1-score of the predictions
      - Kappa: the Cohen Kappa score of the predictions
      - AvgPrecision: The average precision obtained from the
        precision/recall curve
      - LogLoss: The log (cross-entropy) loss 
      - AuROC: The area under the receiver operating curve

    For all results:
      - Count: The number of y (and yhat) values provided

    Parameters
    ----------
    y : array
        The sample labels.
    yhat : array
        The sample predictions.
    classify : bool
        If True: calculate binary classification metrics. y is a list
        of class labels (0/1 or False/True) and yhat is the predicted
        probability of the 1/True label
    metrics : list, optional
        The list of metrics required. The metric names should be a
        case-insensitive subset of the list above. If ``None``, the
        list ``['Count', 'RMSE', 'R2', 'Bias']`` is used for regression
        results and ``['Count', 'Acc', 'F1Score', 'Kappa']`` is used
        for classification results. If 'All', then all regression or
        classification metrics are calculated. The default is None.
    ybar : int or float
        The sample mean. If None, ybar = y.mean(). The default is None.
        Specifying ybar allows calculation of R2 for a sub-sample using
        the sample mean instead of the sub-sample mean. This allows
        comparisons between the R2 for different sub-samples to be
        compared, which can't be done is the sub-sample mean is used.
    threshold : float
        If ``classify``, then the probability threshold to use when
        assigning class labels. The default is 0.5.
    precision : int
        Round calculated statistics to this number of decimal places.

    Returns
    -------
    dict
        The calculated statistics. Keys are the metrics names in the
        same case as specified in the ``metrics`` parameter. 

    """
    def calc_count():
        return y.shape[0]
    
    # Regression Metrics
    def calc_bias():
        bias = np.mean(yhat) - np.mean(y)
        return np.round(bias, precision)
    def calc_rmse():
        rmse = np.sqrt(np.mean(np.square(yhat - y)))
        return np.round(rmse, precision)
    def calc_rmspe():
        rmspe = np.sqrt(np.mean(np.square((yhat - y) / y)))
        return np.round(rmspe, precision)
    def calc_r():
        r = np.corrcoef(y, yhat)
        return np.round(r[0, 1], precision)
    def calc_r2():
        if ybar is None:
            r2 = r2_score(y, yhat)
        else:
            r2 = 1 - (((yhat - y) ** 2).sum() / ((y - ybar) ** 2).sum())
        return np.round(r2, precision)
    def calc_ubrmse():
        bias = calc_bias()
        ubrmse = np.sqrt(np.mean(np.square(yhat - y - bias)))
        return np.round(ubrmse, precision)
    
    # Classification Metrics
    def calc_acc():
        return np.round(accuracy_score(y, preds), precision)
    def calc_f1score():
        if binary:
            average = 'binary'
        else:
            average = 'weighted'
        return np.round(f1_score(y, preds, average=average), precision)
    def calc_kappa():
        return np.round(cohen_kappa_score(y, preds), precision)
    def calc_avg_precision():
        return np.round(average_precision_score(y, yhat), precision)
    def calc_log_loss():
        return np.round(log_loss(y, yhat), precision)
    def calc_auroc():
        return np.round(roc_auc_score(y, yhat), precision)
    def calc_mcc():
        return np.round(matthews_corrcoef(y, yhat), precision)
    def calc_csi():
        if binary:
            average = 'binary'
        else:
            average = 'macro'
        return np.round(jaccard_score(y, yhat, average=average), precision)

    default_regress_metrics = ['Count', 'RMSE', 'R2', 'Bias']
    all_regress_metrics = ['Count', 'RMSE', 'ubRMSE', 'RMSPE', 'R', 'R2', 'Bias']
    default_class_metrics = ['Count', 'Acc', 'F1Score', 'Kappa']
    all_class_metrics = ['Count', 'Acc', 'F1Score', 'Kappa', 'AvgPrecision', 'LogLoss', 'AuROC', 'MCC', 'CSI']
    functions = {'count': calc_count, 'rmse': calc_rmse, 'rmspe': calc_rmspe, 'ubrmse': calc_ubrmse,
                 'bias': calc_bias, 'r': calc_r, 'r2': calc_r2, 'acc': calc_acc,
                 'f1score': calc_f1score, 'kappa': calc_kappa, 'avgprecision': calc_avg_precision,
                 'logloss': calc_log_loss, 'auroc': calc_auroc, 'mcc': calc_mcc, 'csi': calc_csi}
    if classify:
        if (y.min() == 0) and (y.max() == 1) and (yhat.min() == 0) and (yhat.max() == 1):
            binary = True
            preds = (yhat >= threshold)
        else:
            binary = False
            preds = yhat
        if metrics is None:
            metrics = default_class_metrics
        elif isinstance(metrics, str) and metrics.lower() == 'all':
            metrics = all_class_metrics #list(functions.keys())
    else:
        if metrics is None:
            metrics = default_regress_metrics
        elif isinstance(metrics, str) and metrics.lower() == 'all':
            metrics = all_regress_metrics #list(functions.keys())
    try:
        {m: functions[m.lower()] for m in metrics}
    except:
        except_msg = f"Calc_statistics: {metrics} does not define a valid set of metrics" 
        raise ValueError(except_msg) from None
    if len(y) == 0:
        stats_ = {m: 0 if m.lower() == 'count' else np.nan for m in metrics}
    else:
        stats_ = {m: functions[m.lower()]() for m in metrics}
    return stats_


def grouped_results(samples, predicts, grouping_column, target_column, test_list,
                    keys=None, measures='RMSE'):
    """ Calculates statistics for each group of results
    
    Groups the samples and predictions by values of the grouping colum
    and calculates statistics separately for each group of results
    

    Parameters
    ----------
    samples : DataFrame
        A data frame of samples.
    predicts : list - of either DataFrames or lists of DataFrames
        Each dataframe contains a set of test results - the rows are
        the samples and the columns are each test run.
    grouping_column : str
        The samples column to use for grouping.
    target_column : str
        The target column in the samples.
    test_list : list
        A list of test names. Should be the same length as the total
        number of data frames in ``predicts``.
    keys : list, optional
        An optional list of values to select from the grouping column.
        If ``None``, all values in the grouping column are used. The
        default is None.
    measures : str or list of str, optional
        The statictics to return - currently should be a subset of the
        regression defaults. If ``None``, all default regression
        metrics are returned. The default is 'RMSE'.

    Returns
    -------
    results : DataFrame or dict of DataFrames
        If a single measure - a dataframe of the results.
        If a list of measures - a dictionary. Keys are the measures and
        values are a dataframe of the results for the measure.
        The columns of the data frames are the grouping column values
        and the rows are the test names

    """
    temp_results = defaultdict(list)
    if keys is None:
        keys = np.sort(samples[grouping_column].unique())
    if isinstance(predicts[0], list):  # Flatten to a list
        predicts = [item for sublist in predicts for item in sublist]
    for test_preds in predicts:
        for key in keys:
            temp_samples = samples[samples[grouping_column] == key]
            temp_index = temp_samples.index.intersection(test_preds.index)
            if temp_index.size > 0:
                temp_predicts = test_preds.reindex(temp_index)
                temp_samples = temp_samples[target_column].reindex(temp_index)
                temp_stats = [calc_statistics(temp_samples, pred[1])
                              for pred in temp_predicts.iteritems()]
                temp_stats = pd.DataFrame(temp_stats).mean()
            else:
                temp_stats = None #pd.Series({'RMSE': None})
            temp_results[key].append(temp_stats)

    temp_df = pd.concat([pd.DataFrame(temp_results[key], index=test_list) for key in keys],
                        axis=1, keys=keys)
    results = {}
    if measures is None:
        measures = temp_df.columns.get_level_values(1).unique().to_list()
    if isinstance(measures, list):
        for s in measures:
            results[s] = temp_df.xs(s, axis=1, level=1).sort_index(axis=1)
    else:
        results = temp_df.xs(measures, axis=1, level=1).sort_index(axis=1)
    return results                


def bias_variance(model, tests, num_runs, source, all_tests=False):
    """Calculates the model bias and variance.
    

    Parameters
    ----------
    model : string
        Specifies which derived model to use - e.g. 'base' for the base
        model, 'merge10' for the derived model formed by merging the
        last 10 checkpoints.
    tests : list
        A list of test names.
    num_runs : int
        The number of runs in each test.
    source : list
        A list of dataframes containing the predictions for each test.
        A flat list of the runs for all tests, in test then run order
        (i.e. all runs for test 1, then all runs for test 2, etc.).
    all_tests : bool, optional
        If ``True``, the bias and variance is calculated across all
        runs for all tests, as well as the individual tests. The
        default is False.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the bias/variance results.

    """
    df_list = []
    y = []
    test_names = [str(test) for test in tests]
    if all_tests:
        df_all = pd.DataFrame(index=source[0].y.index)
        test_names = test_names + ['All tests']
    for test in range(len(tests)):
        first_run = test * num_runs
        y.append(source[first_run].y)
        df2 = pd.DataFrame(index=y[-1].index)
        for run in range(first_run, first_run + num_runs):
            df2[f'run{run - first_run}'] = source[run][model]
            if all_tests:
                df_all[run] = source[run][model]
        df_list.append(df2)
    if all_tests:
        df_list.append(df_all)
        y.append(source[0].y)

    mse = []
    bias = []
    bias2 = []
    variance = []
    for n, yhat in enumerate(df_list):
        ybar = yhat.mean(axis=1)
        mse.append(((yhat.T - y[n]) ** 2).mean())
        bias.append(ybar - y[n])
        bias2.append(bias[-1] ** 2)
        variance.append(((yhat.T - ybar) ** 2 ).mean())
    
    df = pd.DataFrame([mse_.mean() for mse_ in mse],
                      columns=['MSE'],
                      index=[t for t in test_names])
    df['Bias**2'] = [b.mean() for b in bias2]
    df['Variance'] = [v.mean() for v in variance]
    df['RMSE'] = [np.sqrt(mse_.mean()) for mse_ in mse]
    df['Bias mean'] = [b.mean() for b in bias]
    df['Bias var'] = [b.var() for b in bias]
    df['Bias min'] = [b.min() for b in bias]
    df['Bias max'] = [b.max() for b in bias]
    return df


def samples_with_historical_data(all_samples, all_predictions, site_column='Site',
                                 year_column='Sampling year'):
    """Gets data for samples from sites with historical data.
    
    

    Parameters
    ----------
    all_samples : DataFrame
        Data frame containing the sample data. Should have a row for
        all records in all_predictions, but may have extra rows.
    all_predictions : DataFrame
        Data frame containing the prediction data.
    site_column : Str, optional
        The name of the ``all_samples`` column containing the sampling
        site. The default is 'Site'
    year_column : Str, optional
        The name of the ``all_samples`` column containing the sampling
        year. The default is 'Sampling year'

    Returns
    -------
    yearly_samples : DataFrame
        DESCRIPTION.
    yearly_predicts : DataFrame
        DESCRIPTION.

    """
    temp_samples = all_samples.loc[all_predictions.index]
    yearly_samples = []
    years = temp_samples[year_column].unique()
    for year in years:
        sites = all_samples[all_samples[year_column] < year][site_column].unique()
        yearly_samples.append(temp_samples[temp_samples[site_column].isin(sites)
                                           & (temp_samples[year_column] == year)])
    yearly_samples = pd.concat(yearly_samples)
    yearly_predicts = all_predictions.loc[yearly_samples.index]
    return yearly_samples, yearly_predicts


def sort_file_list(file_list, basename_prefix):
    """Sorts a list of file names.
    
    Sorts a list of file names on the integer part of the name. To sort
    the file names, the file extension and the first part of the file
    name, up to and including the basename_prefix are removed. The
    function assumes the remaining part is an integer. The file names
    are then sorted based on these integer values. For example, calling
    `sort_file_list(['test2.dat', 'test10.dat', 'test1.dat'], 'test')`
    will return `['test1.dat', 'test2.dat', 'test10.dat']`

    Parameters
    ----------
    file_list : List of file names
        The list of file names to be sorted.
    basename_prefix : str
        The basename prefix to remove from the file names.

    Returns
    -------
    List
        The sorted list of file names.

    """
    return sorted(
        file_list,
        key=lambda x: int(os.path.splitext(x)[0].rsplit(basename_prefix, 1)[1] or 0)
        )