"""Model building and evaluation utilities"""

import numpy as np
import os
import pandas as pd

from copy import deepcopy

from analysis_utils import calc_statistics
from data_prep_utils import normalise, denormalise, reweight_source
from lfmc_model import import_model_class, load_model, set_gpus
from results_utils import save_source_stats

   
# =============================================================================
# Model training and evaluation functions
# =============================================================================


def train_model(model, train, val, weights=None):
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
    
    result = model.train(train['X'], train['y'], val['X'], val['y'], weights)
    model.train_result = result

    # Plot the training history
    if model.last_epoch == model.params['epochs']:
        try:
            if model.params['saveTrain']:
                model.plot_train_hist()
            if model.params['saveValidation'] and (model.params['valSize'] > 0):
                model.plot_train_hist(metric='loss')
        except:
            print(f"{model.params['modelName']}: Saving train history plot failed!")
            pass                

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
        _ = model.best_model()                    # Extract the best checkpoint model
        model.merge_models('merge10', 10)         # Merge the last 10 checkpoints into a new model
        model.ensemble_models('ensemble10', 10)   # Create an ensembled model of last 10 checkpoints
        _ = model.best_model(n=10, merge=True)    # Merge the best 10 checkpoint models
    
    # Save models, if required
    save_models = model.params['saveModels']
    if save_models:
        if model.last_epoch == model.params['epochs']:
            prefix = ''
        else:
            prefix = f'epoch{model.last_epoch}'
        if isinstance(save_models, str):
            if save_models.lower() == 'all':
                model.save_to_disk(prefix=prefix)
            else:
                model.save_to_disk(model_name=save_models, prefix=prefix)
        elif isinstance(save_models, list):
            model.save_to_disk(model_list=save_models, prefix=prefix)
        else:
            model.save_to_disk('base', prefix=prefix)


def evaluate_model(model, data, which=None, test_name='test', bias=False, 
                   train_stats=False, save_preds=True):
    """Evaluates a model.
    
    Evaluates a model using the ``data`` dataset. If the ``which``
    parameter is specified, only that child model is run, otherwise
    all child models are run. The model predictions and prediction
    statistics are stored as model attributes ``<test_name>_predicts``
    and ``<test_name>_stats``. If this is not a fold model, the stats
    and (if save_preds) predictions are saved to the model directory.

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
    data_y = data['y'] if data['y'].ndim == 1 else data['y'][model.params['targetColumn']]
    normal = model.params['targetNormalise']
    y = denormalise(data_y, **normal) if normal else data_y
    for model_name in which:
        result_ = model.evaluate(data['X'], data_y, model_name, get_stats=False, plot=False)
        if normal:
            result_['predict'] = denormalise(result_['predict'], **normal)
        if isinstance(bias, pd.Series):
            result_['predict'] = result_['predict'] - bias[model_name]
        result_['stats'] = calc_statistics(y, result_['predict'],
                                           classify=model.params['classify'])
        results[model_name] = result_

    # Create dataframes for predictions and stats
    epoch_key = model.last_epoch == model.params['epochs']
    epoch_key = None if epoch_key else f'epoch{model.last_epoch}'
    if epoch_key:
        model_dir = os.path.join(model_dir, epoch_key)
    all_results = pd.DataFrame({'y': y,
                                **{name: result['predict'] for name, result in results.items()}})
    all_stats = pd.DataFrame([r['stats'] for r in results.values()], index=results.keys())
    all_stats['sampleCount'] = data['y'].shape[0]
    all_stats['trainCount'] = model.train_count
    all_stats['runTime'] = [r['runTime'] for r in results.values()]
    if train_stats:
        all_stats['trainTime'] = model.train_result['runTime']
        all_stats['buildTime'] = model.build_time
        weights = model.weight_counts() 
        all_stats['trainableWeights'] = weights[0]
        all_stats['nonTrainableWeights'] = weights[1]

    # Save the results as model attributes
    if epoch_key:
        getattr(model, f'epoch_{test_name}_predicts')[epoch_key] = all_results
        getattr(model, f'epoch_{test_name}_stats')[epoch_key] = all_stats
    else:
        setattr(model, f'{test_name}_predicts', all_results)
        setattr(model, f'{test_name}_stats', all_stats)
    preds_file = f'{test_name}_predicts.csv'
    stats_file = f'{test_name}_stats.csv'
    if model.params.get('fold') is None:
        # Predictions and stats for a "fold" model are not saved, as these will be merged
        # with the other fold results and stored in the "run" directory.
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if save_preds:  # Save predictions if required. 
            all_results.to_csv(os.path.join(model_dir, preds_file))
        all_stats.to_csv(os.path.join(model_dir, stats_file))

    # If there are multiple data sources, save separate stats for each source
    if all_results.index.nlevels > 1:
        dir_ = model_dir if model.params.get('fold') is None else None
        save_source_stats(model, all_results, which, test_name, dir_, epoch_key)
    

def train_and_evaluate(model, train, val, test, weights=None, pretrained=False):
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
    if model.params['saveFiles'] and not os.path.exists(model.model_dir):
        os.makedirs(model.model_dir)
    if not pretrained:
        train_model(model, train, val, weights)
    model.train_count = train['y'].shape[0]
    debias = model.params.get('debias', False)
    if ((model.params['saveTrain'] is not False)  # i.e. is None or True
        or debias):
        evaluate_model(model, train, test_name='train', train_stats=not pretrained,
                       save_preds=model.params['saveTrain'])
    if debias:
        # Needs to evaluate for data from test sources only
        epoch_key = model.last_epoch == model.params['epochs']
        epoch_key = None if epoch_key else f'epoch{model.last_epoch}'
        if epoch_key:
            results_ = getattr(model, 'epoch_train_predicts')[epoch_key]
        else:
            results_ = getattr(model, 'train_predicts')
        if model.params['testSources']:
            results_ = results_.loc[model.params['testSources']]
        results = {}
        which = ['base'] + list(model.derived_models.keys())
        for model_name in which:
            results[model_name] = calc_statistics(results_.y, results_[model_name],
                                               classify=model.params['classify'])
        bias = pd.DataFrame(results).loc['Bias']
    else:
        bias = None
    if val['y'] is not None:
        evaluate_model(model, val, test_name='val', bias=bias,
                       save_preds=model.params['saveValidation'])
    if test['y'] is not None:
        evaluate_model(model, test, bias=bias)
    else:
        model.test_predicts = None
        model.test_stats = None


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

    def normalise_input(train_data, val_data, test_data, input_name, normal_params,
                        model_dir=None, save_params=False, update_params=True, columns=False):
        if not update_params:
            normal_params = deepcopy(normal_params)

        if columns:
            train_data = train_data.copy()
            train_data[columns[0]], params = normalise(
                train_data[columns[0]],
                **normal_params,
                input_name=input_name,
                model_dir=model_dir,
                save_params=save_params,
                return_params=True)
            normal_params.update(params)
            for col in columns[1:]:
                train_data[col] = normalise(train_data[col], **normal_params)
        else:
            train_data, params = normalise(
                train_data,
                **normal_params,
                input_name=input_name,
                model_dir=model_dir,
                save_params=save_params,
                return_params=True)
            normal_params.update(params)

        if val_data is not None:
            if columns:
                val_data = val_data.copy()
                for col in columns:
                    val_data[col] = normalise(val_data[col], **normal_params)
            else:
                val_data = normalise(val_data, **normal_params)

        if test_data is not None:
            if columns:
                test_data = test_data.copy()
                for col in columns:
                    test_data[col] = normalise(test_data[col], **normal_params)
            else:
                test_data = normalise(test_data, **normal_params)
        return train_data, val_data, test_data
    
    def normalise_all_inputs(train_data, val_data, test_data, model_params, update_params=True):
        model_dir = model_params['modelDir']
        save_params = model_params['saveModels'] is not False
        for input_name, input_parms in model_params['inputs'].items():
            normal = input_parms.get('normalise', {}) or {}
            (train_data['X'][input_name],
                val_data['X'][input_name],
                test_data['X'][input_name]) = \
                normalise_input(train_data['X'][input_name],
                                val_data['X'][input_name],
                                test_data['X'][input_name],
                                input_name, normal, model_dir, save_params, update_params)
        normal = model_params.get('targetNormalise', {}) or {}
        if train_data['y'].ndim == 1:
            columns = None
        else:
            columns = list(train_data['y'].columns)
            if model_params.get('multiTarget', False):
                columns = columns[0:1]
        train_data['y'], val_data['y'], test_data['y'] = \
            normalise_input(train_data['y'], val_data['y'], test_data['y'], 'target',
                            normal, model_dir, save_params, update_params, columns)

    def normalise_data(train_data, val_data, test_data, model_params, source=None):
        if isinstance(model_params['epochs'], int): #source is None:
            normalise_all_inputs(train_data, val_data, test_data, model_params)
        else:
            if source is not None:
                # Normalise the main source and test data. Update normalise parameters so
                # other sources are normalised using the same parameters.
                normalise_all_inputs(train_data[source], val_data[source], test_data, model_params)
                test_source = source
            else:
                test_source = model_params['testSources'][0]
                
            # Normalise other sources
            for src in model_params['epochs'].keys():
                if src != source:  # Skip the main source
                    if src == test_source:  # Normalise the test data
                        test_ = test_data
                        save_ = True        # Save updated normalisation params for any future tests
                    else:  # No test data for this source to normalise
                        test_ = {'X': {x: None for x in model_params['inputs'].keys()},
                                 'y': None}
                        save_ = False
                    normalise_all_inputs(train_data[src], val_data[src], test_,
                                         model_params, update_params=save_)

    def augment_data(train_data, val_data, test_data, model_params):
        aug = model_params['auxAugment']
        if not aug or ('aux' not in model_params['dataSources']):
            return    # No auxiliary augmentation required
        if isinstance(model_params['epochs'], int):
            input_list = [train_data, val_data, test_data]
        else:
            input_list = list(train_data.values()) + list(val_data.values()) + [test_data]
        for input_ in input_list:
            if 'aux' not in input_['X'].keys():
                continue   # Input has no auxiliary data to augment
            for input_name, input_data in input_['X'].items():
                if (input_data is not None) and (len(input_data.shape) == 3):
                    if (aug is True) or (isinstance(aug, list) and input_name in aug):
                        input_['X']['aux'] = np.concatenate(
                            [input_['X']['aux'], input_data[:, -1, :]], axis=1)
                    elif isinstance(aug, dict) and input_name in aug.keys():
                        offset = aug[input_name] or 1
                        input_['X']['aux'] = np.concatenate(
                            [input_['X']['aux'], input_data[:, -offset, :]], axis=1)
            
    def load_pretrained_model(model_params): #, inputs, targets):
        diagnostics = model_params['diagnostics']
        pretrained_dir = model_params['pretrainedModel']
        test = model_params['test']
        test = '' if test is None else f'test{test}'
        run = model_params['run']
        run = '' if run is None else f'run{run}'
        fold = model_params['fold'] or ''
        fold = '' if fold is None else f'fold_{fold}'
        year_fold = fold[:fold.rfind('_')] if fold else ''
        dir_, last = os.path.split(pretrained_dir)
        if last.startswith('fold'):
            dir_list = [pretrained_dir]
        elif last.startswith('run'):
            dir_list = [os.path.join(pretrained_dir, fold),
                        os.path.join(pretrained_dir, year_fold),
                        pretrained_dir]
        elif last.startswith('test'):
            dir_list = [os.path.join(pretrained_dir, run, fold),
                        os.path.join(pretrained_dir, run, year_fold),
                        os.path.join(pretrained_dir, run),
                        os.path.join(pretrained_dir, fold),
                        os.path.join(pretrained_dir, year_fold),
                        pretrained_dir]
        else:
            if test and os.path.exists(os.path.join(pretrained_dir, test)):
                dir_list = [os.path.join(pretrained_dir, test, run, fold),
                            os.path.join(pretrained_dir, test, run, year_fold),
                            os.path.join(pretrained_dir, test, run),
                            os.path.join(pretrained_dir, test, fold),
                            os.path.join(pretrained_dir, test, year_fold),
                            os.path.join(pretrained_dir, test),]
            else:
                dir_list = [os.path.join(pretrained_dir, run, fold),
                            os.path.join(pretrained_dir, run, year_fold),
                            os.path.join(pretrained_dir, run),
                            os.path.join(pretrained_dir, fold),
                            os.path.join(pretrained_dir, year_fold),
                            pretrained_dir]
        for dir_ in dir_list:
            try:
                if os.path.exists(dir_):
                    model = load_model(dir_)
                    model_params['pretrainedModel'] = dir_
                    if diagnostics:
                        print(f'{test} {run} {fold}: Pre-trained model loaded from {dir_}')
                    break
            except:
                pass
        else:
            return None
        # update model parameters
        old_params = deepcopy(model.params)
        model_params['blocks'] = old_params['blocks']
        if model_params.get('commonNormalise', True):
            new_inputs = model_params['inputs']
            old_inputs = old_params['inputs']
            for input_ in model_params['inputs'].keys():
                new_inputs[input_]['normalise'] = old_inputs[input_]['normalise']
            model_params['targetNormalise'] = old_params['targetNormalise']
        source_size = old_params.get('trainSize', 0)
        
        # work-around for missing size from conus_base_models
        if (source_size == 0) and ('conus_base_models' in model_params['pretrainedModel']):
            source_size = {'fold_2014': 46031,
                           'fold_2015': 51859,
                           'fold_2016': 57336,
                           'fold_2017': 62466,
                          }[fold if year_fold == 'fold' else year_fold]
            if diagnostics:
                print(f"Setting source size for fold {fold} to {source_size}")
            
        return model, source_size

    def train_for_source(model, epoch_step, last_epoch, train, val, test): #, weights=None):
        if model.params['reweightSource']:
            source = model.params['reweightSource'][0]
            target = model.params['reweightSource'][1]
            weights = reweight_source(train['y'], source, target)
        else:
            weights = None

        if epoch_step:
            first_save_epoch = model.last_epoch + epoch_step
            for epoch_stop in range(first_save_epoch, last_epoch, epoch_step):
                model.last_epoch = epoch_stop
                train_and_evaluate(model, train, val, test, weights)
                if model_params['diagnostics']:
                    print(f"Trained to epoch {epoch_stop}")

        model.last_epoch = last_epoch
        train_and_evaluate(model, train, val, test, weights)
        if model_params['diagnostics']:
            print(f"Training finished at epoch {last_epoch}")
        
##########
    
    gpu = getattr(train_test_model, 'gpu', None)
    if isinstance(gpu, (int, list)):
        model_params['gpuDevice'] = gpu
    set_gpus(model_params['gpuDevice'])

    if model_params.get('deduplicate'):
        train = deduplicate(train)
    model_dir = model_params['modelDir']
    model_params['saveFiles'] = (model_params['fold'] is None
                                 or model_params['saveTrain']
                                 or (model_params['saveValidation'] and model_params['valSize'])
                                 or model_params['saveModels'] is not False) #or True
    if model_params['saveFiles'] and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    epochs = model_params['epochs']
    pretrained = model_params.get('pretrainedModel') is not None
    if pretrained:
        # Only handles a single source
        model, source_size = load_pretrained_model(model_params) #, train['X'], train['y'])
        model_params['trainSize'] = train['y'].shape[0]
        normalise_data(train, val, test, model_params)
        augment_data(train, val, test, model_params)
        model.train_data = train
        model.val_data = val
        model.test_data = test
        model.setup_finetune(model_params, train['X'], train['y'].shape[0], source_size)
    elif isinstance(epochs, int):
        model_params['trainSize'] = train['y'].shape[0]
        normalise_data(train, val, test, model_params)
        augment_data(train, val, test, model_params)
        model = import_model_class(model_params['modelClass'])(
            model_params, inputs=train['X'])
    else:
        source = list(epochs.keys())[0] if model_params.get('commonNormalise', True) else None
        normalise_data(train, val, test, model_params, source)
        augment_data(train, val, test, model_params)
        model = import_model_class(model_params['modelClass'])(
            model_params, inputs=train[list(epochs.keys())[0]]['X'])
    
    if model_params['plotModel']:
        outdir = model_dir.rstrip('/\\')
        if os.path.basename(outdir).startswith('fold'):
            outdir = os.path.dirname(outdir)
        if os.path.basename(outdir).startswith('run'):
            outdir = os.path.dirname(outdir)
        model.plot(dir_name=outdir)
    
    epoch_step = model_params['evaluateEpochs']
    if epoch_step or isinstance(epochs, dict):
        model.epoch_test_predicts = {}
        model.epoch_test_stats = {}
        model.epoch_train_predicts = {}
        model.epoch_train_stats = {}
        model.epoch_val_predicts = {}
        model.epoch_val_stats = {}

    try:
        model.last_epoch = 0
        if pretrained and epoch_step:
            train_and_evaluate(model, train, val, test, pretrained=True)
        if isinstance(epochs, int):
            train_for_source(model, epoch_step, epochs, train, val, test) #, weights)

        else:
            model.params['epochs'] = np.sum(list(epochs.values()))
            test_ = {'y': None}
            for source, epochs in epochs.items():
                test_source = source in model_params['testSources']
                test_ = test if epoch_step or test_source else {'y': None}
                if model_params.get('transferModel'):
                    if model.last_epoch == 0:
                        first_size = train[source]['y'].shape[0]
                    else:
                        this_size = train[source]['y'].shape[0]
                        model.transfer(this_size, first_size)
                print(f"Training for {epochs} epochs with {source} data")
                last_epoch = model.last_epoch + epochs
                train_for_source(model, epoch_step, last_epoch, train[source], val[source], test_)
            if test_['y'] is None and test['y'] is not None:
                evaluate_model(model, test)

    except Exception as e:
        e.args = (f"{model.params['modelName']}", *e.args)
        raise
    finally:
        model.clear_model()
        try:
            delattr(model, 'train_data')
            delattr(model, 'val_data')
            delattr(model, 'test_data')
        except:
            pass
        if model_params['diagnostics']:
            print(f"{model.params['modelName']} Finished!")
    return model
