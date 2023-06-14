"""Results display utilities."""
import numpy as np
import os
import pandas as pd
from IPython.display import display_html 

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from analysis_utils import plot_results, bias_variance


def print_heading(text, line_char='-', line_before=False, line_after=True,
                  blank_before=0, blank_after=1, indent=0):
    print_list = [''] * blank_before
    if line_before:
        print_list.append(''.join([' ' * indent, line_char * len(text)]))
    print_list.append(''.join([' ' * indent, text]))
    if line_after:
        print_list.append(''.join([' ' * indent, line_char * len(text)]))
    print_list.extend([''] * blank_after)
    print('\n'.join(print_list))


def display_frames(frames, captions=None, precision=3, separator="\xa0" * 10):
    """Displays a list of dataframes.
    
    Parameters
    ----------
    frames : list
        The list of dataframes to display.
    captions : list, optional
        A list of captions for the dataframes. If any captions are
        provided there must be a caption for each dataframe. The
        default is None.
    precision : int, optional
        Precision for floating point numbers in the dataframes. The
        default is 3.
    separator : string, optional
        Dataframe separator. The default is "\xa0" * 10 (10 spaces).

    Returns
    -------
    string
        HTML to display the dataframes in a notebook.

    """
    if captions is None:
        captions = [None] * len(frames)
    elif len(captions) < len(frames):
        captions = captions + [None] * len(frames)
    styles = [df.style if type(df) is pd.DataFrame else df for df in frames]
    styles = [df.set_table_attributes("style='display:inline'").format(precision=precision)
              .set_caption(captions[c]) for c, df in enumerate(styles)]
    for df in styles:
        for row in df.index:
            for column in df.columns:
                val = df.data.loc[row, column]
                if type(val) == np.float64 and abs(val) < 10**-precision and val != 0.0:
                    df = df.format({column: "{:.1e}"}, subset=(row, column))
    allStyles = styles[0]._repr_html_()
    for style in styles[1:]:
        allStyles += separator + style._repr_html_()
    return display_html(allStyles, raw=True)


def bold(val, sigLevel):
    """
    Takes a scalar and returns the css property 'font-weight: bold' if
    the value is less than the 'sigLevel' value, 'font-weight: normal'
    otherwise.
    """
    weight = 'bold' if val < sigLevel else 'normal'
    return 'font-weight: %s' % weight


def bold_col(val, sigLevel):
    """
    Takes a series that includes a p-value and returns a series of css
    properties 'font-weight: bold' if the p-value is less than
    'sigLevel', 'font-weight: normal' otherwise.
    """
    weight = 'bold' if val['p-value'] < sigLevel else 'normal'
    return pd.Series([f'font-weight: {weight}'] * len(val), index=val.index)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    """Truncates a matplotlib colormap
    

    Parameters
    ----------
    cmap : colormap
        The input color map.
    minval : float, optional
        The lower bound for the output colormap. The default is 0.0.
    maxval : float, optional
        The upper bound for the output colormap. . The default is 1.0.
    n : int, optional
        Number of fixed points on the colormap. The default is -1 (use
        the number on the input colormap).

    Returns
    -------
    new_cmap : colormap
        The new colormap.

    """
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_all_results(all_results, all_stats, model_dir, prefix=''):
    """Plot results for all models
    
    Parameters
    ----------
    all_results : Data Frame
        A data frame contains the results to plot. The first column
        should be named ``y`` and contain the labels. The other columns
        should be named with the model name.
    all_stats : Data Frame
        A data frame containing the evaluation statistics for each
        model. The index should be the model names and the columns the
        statistic names.
    model_dir : str
        The name of the directory where the plots should be saved.

    Returns
    -------
    None.

    """
    for y in all_results.columns.drop('y'):
        fig = plot_results(f'{y} Results', all_results.y, all_results[y], all_stats.loc[y])
        fig.savefig(os.path.join(model_dir, prefix + y + '.png'), dpi=300)
        plt.close(fig)


def results_summary(model, tests, ensembles, means, variances=None, std_devs=None, precision=2):
    """Summarizes a set of model results.
    

    Parameters
    ----------
    model : string
        Specifies which derived model to use - e.g. 'base' for the base
        model, 'merge10' for the derived model formed by merging the
        last 10 checkpoints.
    tests : list
        A list of test names.
    ensembles : list
        A list of dataframes (one for each test) containing the model
        statistics generated by creating an ensemble containing all the
        runs in a test. If ``None`` ensembled results are not generated.
    means : list
        A list of dataframes (one for each test) containing the means
        of the model statistics.
    variances : list, optional
        A list of dataframes (one for each test) containing the
        variances of the model statistics. The default is None -
        variances are not displayed.
    std_devs : list, optional
        A list of dataframes (one for each test) containing the
        standard deviations of the model statistics. The default is
        None - standard deviations are not displayed.
    precision : int, optional
        The floating-point precision to use when displaying the
        dataframes. The default is 2.

    Returns
    -------
    None.

    """
    test_list = [str(test) for test in tests]
    separate_index = pd.Series(test_list).apply(len).max() > 20
    if separate_index:
        index = [f'Test{t}' for t in range(len(tests))]
    else:
        index = test_list
    tdf = pd.DataFrame(test_list, index=index, columns=['Test'])
    df_list = []
    captions = []
    if ensembles:
        edf = pd.DataFrame([e.loc[model] for e in ensembles], index=index)
        df_list.append(edf)
        captions.append('Ensembles')
    mdf = pd.DataFrame([m.loc[model] for m in means], index=index)
    df_list.append(mdf)
    captions.append('Means')
    if variances:
        vdf = pd.DataFrame([v.loc[model] for v in variances], index=index)
        df_list.append(vdf)
        captions.append('Variances')
    if std_devs:
        sdf = pd.DataFrame([s.loc[model] for s in std_devs], index=index)
        df_list.append(sdf)
        captions.append('Standard Deviations')
    if len(df_list) == 1:
        df_list[0].set_index(pd.Index(test_list), inplace=True)
        heading = f'{captions[0]} of {model} model test results'
    else:
        if separate_index:
            df_list = [tdf] + df_list
            captions = [''] + captions
        heading = f'{", ".join([x for x in captions[:-1] if x])} and {captions[-1]} of {model} ' \
                  'model test results'
    print("\n" + heading)
    print("-" * len(heading))
    display_frames(df_list, captions, precision=precision)


def display_bias_variance(model, tests, num_runs, source, all_tests=False, heading=None, precision=3):
    """Displays the model bias and variance.
    

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
    heading : str, optional
        Display heading. If None, ``Model: {model}`` is used. The
        default is None.
    precision : int, optional
        The floating-point precision to use when displaying the
        dataframe. The default is 3.

    Returns
    -------
    None.

    """
    df = bias_variance(model, tests, num_runs, source, all_tests=False)
    if heading is None:
        heading = f"Model: {model}"
#    print("\n" + heading)
#    print("-" * len(heading))
    print_heading(heading, blank_before=1, blank_after=0)
    display_frames([df], precision=precision)