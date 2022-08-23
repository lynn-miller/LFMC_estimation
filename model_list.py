"""ModelList class."""

import os
import json
import glob
import pandas as pd

from model_parameters import ModelParams
from lfmc_model import load_model


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
    def sort_file_list(file_list, basename_prefix):
        return sorted(
            file_list,
            key=lambda x: int(os.path.splitext(x)[0].rsplit(basename_prefix, 1)[1] or 0)
            )
        
    @classmethod
    def load_folds(cls, model_dir, epoch=None):
        """Loads the folds for a run
        
        Creates a ModelList and populates it by loading the k-fold
        models for a run. Returns ``False`` if the run did not use
        k-fold splits (i.e. ``model_dir`` does not contains fold
        directories).

        Parameters
        ----------
        model_dir : str
            The name of the directory containing the model run.

        Returns
        -------
        ModelList
            The loaded run - a ModelList of folds.
        """
        model_file = 'model_params.json'
#        fold_list = cls.sort_file_list(glob.glob(os.path.join(model_dir, 'fold*')), 'fold')
        fold_list = sorted(glob.glob(os.path.join(model_dir, 'fold*')))
        if len(fold_list) == 0:
            return False
        try:
            with open(os.path.join(model_dir, model_file), 'r') as f:
                model_params = ModelParams(f)
        except:
            raise FileNotFoundError(f'Model parameters missing: {model_file}')
        model_name = model_params['modelName']
        model_list = cls(model_name, model_dir)
        model_list.params = model_params
        if epoch is None:
            dir_name = model_dir
        else:
            dir_name = os.path.join(model_dir, f'epoch{epoch}')
            model_list.params['modelDir'] = dir_name
            model_list.params['epochs'] = epoch
        try:
            model_list.all_stats = pd.read_csv(os.path.join(dir_name, 'predict_stats.csv'),
                                               index_col=0)
            model_list.all_results = pd.read_csv(os.path.join(dir_name, 'predictions.csv'),
                                                 index_col=0)
        except:
            pass
        for fold_dir in fold_list:
            model_list.append(load_model(fold_dir, epoch=epoch))
        return model_list

    @classmethod
    def load_model_set(cls, model_dir, num_runs=None, epoch=None):
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

        Returns
        -------
        ModelList
            The loaded models - a ModelList of runs.
        """
        model_file = 'model_params.json'
        try:
            with open(os.path.join(model_dir, model_file), 'r') as f:
                model_params = ModelParams(f)
        except:
            raise FileNotFoundError(f'Model parameters missing: {model_file}')
        run_list = cls.sort_file_list(glob.glob(os.path.join(model_dir, 'run*')), 'run')
        if num_runs:
            run_list = run_list[:num_runs]

        if len(run_list) == 0: # A single run, so load the model
            if model_params['splitFolds'] <= 1:
                return load_model(model_dir, epoch=epoch)
            else:
                return cls.load_folds(model_dir, epoch=epoch)
 
        model_name = model_params['modelName']
        model_list = cls(model_name, model_dir)
        model_list.params = model_params
        if epoch is None:
            dir_name = model_dir
        else:
            dir_name = os.path.join(model_dir, f'epoch{epoch}')
            model_list.params['modelDir'] = dir_name
            model_list.params['epochs'] = epoch
        try: # Load predictions and statistics files if they exist
            model_list.run_stats = pd.read_csv(os.path.join(dir_name, 'stats_all.csv'),
                                               index_col=[0, 1])
            model_list.means = pd.read_csv(os.path.join(dir_name, 'stats_means.csv'),
                                           index_col=0)
            model_list.variances = pd.read_csv(os.path.join(dir_name, 'stats_vars.csv'),
                                               index_col=0)
            model_list.all_stats = pd.read_csv(os.path.join(dir_name, 'ensemble_stats.csv'),
                                               index_col=0)
            model_list.all_results = pd.read_csv(os.path.join(dir_name, 'ensemble_preds.csv'),
                                                 index_col=0)
        except:
            model_list.all_results = None
        for run_dir in run_list:
            if model_params['splitFolds'] <= 1:
                model_list.append(load_model(run_dir, epoch=epoch))
            else:
                model_list.append(cls.load_folds(run_dir, epoch=epoch))
        return model_list

    @classmethod
    def load_experiment(cls, model_dir, experiment_only=True, num_tests=None, epoch=None):
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
        param_files = cls.sort_file_list(param_files, 'experiment')
        if len(param_files) == 0:
            if experiment_only:
                raise FileNotFoundError(f'Experiment parameters missing: {experiment_files}')
            else: # Not an experiment, so just load the test
                models = cls.load_model_set(model_dir, epoch=epoch)
                if isinstance(models, ModelList):
                    models.experiment = False
                return models
        with open(param_files[-1], 'r') as f:
            experiment = json.load(f)

        # Load the global model parameters
        model_files = 'model_params*.json'
        param_files = glob.glob(os.path.join(model_dir, model_files))
        param_files = cls.sort_file_list(param_files, 'params')
        if len(param_files) == 0:
            raise FileNotFoundError(f'Model parameters missing: {model_files}')
        with open(param_files[-1], 'r') as f:
            model_params = ModelParams(f)
        model_name = model_params['modelName']

        model_list = cls(model_name, model_dir)
        model_list.experiment = experiment
        model_list.params = model_params
        test_dirs = cls.sort_file_list(glob.glob(os.path.join(model_dir, 'test*')), 'test')
        if num_tests:
            test_dirs = test_dirs[:num_tests]
        for test_dir in test_dirs:
            try:
                model_list.append(cls.load_model_set(test_dir, epoch=epoch))
            except:
                break
        return model_list
