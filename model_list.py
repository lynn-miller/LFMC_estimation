"""ModelList class."""

import os
import glob
import pandas as pd

from model_parameters import ModelParams, ExperimentParams
from lfmc_model import load_model, sort_file_list, load_files, load_epoch_files


class ModelList(list):
    """A list of models or ModelLists
    
    Adds a name and model_dir attribute to a list and allows related
    models to be grouped hierarchically - e.g. if they represent a set
    of tests, and each test is run several times. 
    """
    
    def __init__(self, name, model_dir):
        self.name = name
        self.model_dir = model_dir
        super(ModelList, self).__init__()

    @staticmethod
    def load_epochs(model, model_dir, multi_source=False, pred_type='single'):
        model.epoch_test_predicts = {}
        model.epoch_test_stats = {}
        if pred_type == 'single':
            for test_name in ['test', 'val', 'train']:
                load_epoch_files(model, model_dir, test_name, multi_source=multi_source)
                load_epoch_files(model, model_dir, f'fold_{test_name}', multi_source=multi_source)
        else:
            load_epoch_files(model, model_dir, 'ensemble', 'test', multi_source=multi_source)
        for test_name in ['test', 'val', 'train']:
            load_epoch_files(model, model_dir, test_name, multi_source=multi_source)
            load_epoch_files(model, model_dir, f'fold_{test_name}', multi_source=multi_source)

    @classmethod
    def load_model_run(cls, model_dir, epoch=None, get_folds=True):
        """Loads the folds for a run
        
        Creates a ModelList and loads the run results. Optionally
        loads any epoch results and/or populates the ModelList by
        loading any fold models for the run. 

        Parameters
        ----------
        model_dir : str
            The name of the directory containing the model run.
        epoch : int
            The epoch to load. If ``None``, all epochs are loaded. If
            anything else and the directory ``epoch{epoch}`` exists in
            ``model_dir`` then that epoch is loaded as if it were the
            final epoch. Otherwise just the final epoch is loaded. The
            default is None.
        get_folds : bool
            If ``True`` then any folds for the run are loaded. If
            ``False`` then the folds are not loaded. The default is
            True.

        Returns
        -------
        ModelList or LfmcModel
            The loaded run - the model or a ModelList of folds.
        """
        model_file = 'model_params.json'
        fold_list = glob.glob(os.path.join(model_dir, 'fold*'))
        fold_list = sorted([d for d in fold_list if os.path.isdir(d)])
        if len(fold_list) == 0:
            return load_model(model_dir, epoch=epoch)
        try:
            with open(os.path.join(model_dir, model_file), 'r') as f:
                model_params = ModelParams(f)
        except:
            raise FileNotFoundError(f'Model parameters missing: {model_file}')
        model_name = model_params['modelName']
        model_list = cls(model_name, model_dir)
        model_list.params = model_params
        multi_source = model_params.get('sourceNames', None) is not None
        if epoch is None:
            dir_name = model_dir
            if model_params['evaluateEpochs']:
                cls.load_epochs(model_list, model_dir, multi_source)
        else:
            dir_name = os.path.join(model_dir, f'epoch{epoch}')
            if os.path.exists(dir_name):
                dir_name = os.path.join(model_dir, f'epoch{epoch}')
                model_list.params['modelDir'] = dir_name
                model_list.params['epochs'] = epoch
            else: 
                dir_name = model_dir
        for test_name in ['test', 'val', 'train']:
            load_files(model_list, dir_name, test_name)
            load_files(model_list, dir_name, f'fold_{test_name}')
        if get_folds:
            for fold_dir in fold_list:
                if os.path.exists(os.path.join(fold_dir, model_file)):
                    model_list.append(load_model(fold_dir, epoch=epoch))
        return model_list

    @classmethod
    def load_model_set(cls, model_dir, num_runs=None, epoch=None, get_folds=True):
        """Loads a set of models
        
        Creates a ModelList and populates it by loading all models in
        the model set (test or ensemble). If model_dir does not contain
        run directories, assumes the directory is for a single model
        and calls the load_model function to load it.

        Parameters
        ----------
        model_dir : str
            The name of the directory containing the models.
        num_runs : int, optional
            The number of models to load. If falsy (e.g. ``None``), all
            models are loaded. Ignored if model_dir is for a single
            run. The default is None.
        epoch : int
            The epoch to load. If ``None``, all epochs are loaded. If
            anything else and the directory ``epoch{epoch}`` exists in
            ``model_dir`` then that epoch is loaded as if it were the
            final epoch. Otherwise just the final epoch is loaded. The
            default is None.
        get_folds : bool
            If ``True`` then all folds of each run in the model_set are
            loaded. If ``False`` then the folds are not loaded. The 
            default is True.

        Returns
        -------
        ModelList or LfmcModel
            The loaded models - the model or a ModelList of runs.
        """
        model_file = 'model_params.json'
        try:
            with open(os.path.join(model_dir, model_file), 'r') as f:
                model_params = ModelParams(f)
        except:
            raise FileNotFoundError(f'Model parameters missing: {model_file}')
        run_list = sort_file_list(glob.glob(os.path.join(model_dir, 'run*')), 'run')
        if num_runs:
            run_list = run_list[:num_runs]
        has_no_folds = (model_params['splitFolds'] or 0) + (model_params['yearFolds'] or 0) <= 1

        if len(run_list) == 0: # A single run, so load the model
            if has_no_folds:
                return load_model(model_dir, epoch=epoch)
            else:
                return cls.load_model_run(model_dir, epoch=epoch, get_folds=get_folds)
 
        model_name = model_params['modelName']
        model_list = cls(model_name, model_dir)
        model_list.params = model_params
        multi_source = model_params.get('sourceNames', None) is not None
        if epoch is None:
            dir_name = model_dir
            if model_params['evaluateEpochs']:
                cls.load_epochs(model_list, model_dir, multi_source, pred_type='ensemble')
        else:
            dir_name = os.path.join(model_dir, f'epoch{epoch}')
            if os.path.exists(dir_name):
                dir_name = os.path.join(model_dir, f'epoch{epoch}')
                model_list.params['modelDir'] = dir_name
                model_list.params['epochs'] = epoch
            else: 
                dir_name = model_dir
        try: # Load predictions and statistics files if they exist
            model_list.run_stats = pd.read_csv(os.path.join(dir_name, 'stats_all.csv'),
                                               index_col=[0, 1])
            model_list.means = pd.read_csv(os.path.join(dir_name, 'stats_means.csv'),
                                           index_col=0)
            model_list.variances = pd.read_csv(os.path.join(dir_name, 'stats_vars.csv'),
                                               index_col=0)
            model_list.test_stats = pd.read_csv(os.path.join(dir_name, 'ensemble_stats.csv'),
                                                index_col=0)
            pred_index = (0, 1) if model_params.get('sourceNames', None) else 0
            model_list.test_predicts = pd.read_csv(os.path.join(dir_name, 'ensemble_preds.csv'),
                                                   index_col=pred_index)
        except:
            model_list.test_predicts = None
        for run_dir in run_list:
            if has_no_folds:
                model_list.append(load_model(run_dir, epoch=epoch))
            else:
                model_list.append(cls.load_model_run(run_dir, epoch=epoch, get_folds=get_folds))
        return model_list

    @classmethod
    def load_experiment(cls, model_dir, experiment_only=True,
                        num_tests=None, epoch=None, get_folds=True):
        """Loads an experiment
        
        Creates a ModelList and populates it by loading the completed
        tests in an experiment. If model_dir is not an experiment
        directory (does not contain a file called ``experiment.json``),
        assumes the directory is a test and calls the load_model_set
        class method to load it.

        Parameters
        ----------
        model_dir : str
            The name of the directory containing the experiment.
        experiment_only: bool
            True: model_dir must be an experiment directory.
            False: model_dir can be an experiment or test directory.
        num_tests : int, optional
            The number of tests to load. If falsy (e.g. ``None``), all
            tests are loaded. Ignored if ``model_dir`` contains a test
            instead of an experiment. The default is None.

        Returns
        -------
        ModelList
            The loaded experiment - a ModelList of tests.
        """
        # Load the experiment parameters
        experiment_files = 'experiment*.json'
        param_files = glob.glob(os.path.join(model_dir, experiment_files))
        param_files = sort_file_list(param_files, 'experiment')
        if len(param_files) == 0:
            if experiment_only:
                raise FileNotFoundError(f'Experiment parameters missing: {experiment_files}')
            else: # Not an experiment, so just load the test
                models = cls.load_model_set(model_dir, epoch=epoch, get_folds=get_folds)
                if isinstance(models, ModelList):
                    models.experiment = False
                return models
        with open(param_files[-1], 'r') as f:
            experiment = ExperimentParams(f)

        # Load the global model parameters
        model_files = 'model_params*.json'
        param_files = glob.glob(os.path.join(model_dir, model_files))
        param_files = sort_file_list(param_files, 'params')
        if len(param_files) == 0:
            raise FileNotFoundError(f'Model parameters missing: {model_files}')
        with open(param_files[-1], 'r') as f:
            model_params = ModelParams(f)
        model_name = model_params['modelName']

        model_list = cls(model_name, model_dir)
        model_list.experiment = experiment
        model_list.params = model_params
        test_dirs = sort_file_list(glob.glob(os.path.join(model_dir, 'test*')), 'test')
        if num_tests:
            test_dirs = test_dirs[:num_tests]
        for test_dir in test_dirs:
            try:
                model_list.append(cls.load_model_set(test_dir, epoch=epoch, get_folds=get_folds))
            except:
                break
        return model_list
