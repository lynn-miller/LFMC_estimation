"""Result analysis utilities."""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats
from scipy.interpolate import interpn

from sklearn.metrics import r2_score


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
    FONT_SIZE = 8
    TITLE_SIZE = 10
    TEXT_OFFSET = 5
    REGRESS_OFFSET = 25
    MARKER_SIZE = 5
    
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
            tl = f"$RMSE: {tl['RMSE']:.2f}%$\n$ubRMSE: {tl['ubRMSE']:.2f}%$\n$R^2: {tl['R2']:.2f}$"
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

    # Add other plot components
    if diagonal:
        ax.plot([lower, upper], [lower, upper], '--', color=(0.5, 0.5, 0.5))
    ax.grid(show_grid)
    add_text_boxes(ax, top_text, bottom_text, lower, upper)
    if regress:
        add_regression_line(ax, y, yhat, lower, upper, regress, regress_eqn)
    label_plot(ax, fig_name, lower, upper)

    # Return the axes or figure
    if fig is None:
        return ax
    else:
        return fig


def calc_statistics(x, y, precision=2):
    """Calculate model evaluation statistics
    
    Calculates the following statistics:
      - Bias: The difference between the mean prediction and the mean
        label
      - RMSE: The root mean squared error of the predictions
      - ubRMSE (Unbiased RMSE): The RMSE obtained if each prediction is
        adjusted by the Bias
      - R: The correlation coefficient
      - R2 (R-squared): The percent of variance explained

    Parameters
    ----------
    x : array
        The sample labels.
    y : array
        The sample predictions.
    precision : int
        Round calculated statistics to this number of decimal places 

    Returns
    -------
    dict
        The calculated statistics.

    """
    bias = np.mean(y) - np.mean(x)
    bias = np.round(bias, precision)
    rmse = np.sqrt(np.mean(np.square(y - x)))
    rmse = np.round(rmse, precision)
    r = np.corrcoef(x, y)
    r = np.round(r[0, 1], precision)
    r2 = r2_score(x, y)
    r2 = np.round(r2, precision)
    ubrmse = np.sqrt(np.mean(np.square(y - x - bias)))
    ubrmse = np.round(ubrmse, precision)
    return {'Bias': bias, 'R':r, 'R2':r2, 'RMSE': rmse, 'ubRMSE': ubrmse}


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
