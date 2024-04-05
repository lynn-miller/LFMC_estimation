"""Deep Learning model used for LFMC estimation."""

import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import string
import tensorflow as tf
import tensorflow.keras as keras
import time

from model_parameters import ModelParams
from analysis_utils import calc_statistics, plot_results
from tf_custom import import_transfer_method
import tf_custom as custom


def set_gpus(gpus):
    if isinstance(gpus, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in gpus])
    elif gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)


def sort_file_list(file_list, basename_prefix):
    return sorted(
        file_list,
        key=lambda x: int(os.path.splitext(x)[0].rsplit(basename_prefix, 1)[1] or 0)
        )


def load_files(model, dir_name, file_type, attr_name=None, multi_source=False):
    try:
        attr_name = attr_name or file_type
        stats_file = os.path.join(dir_name, f'{file_type}_stats.csv')
        setattr(model, f'{attr_name}_stats', pd.read_csv(stats_file, index_col=0))
        pred_index = (0,1) if multi_source else 0
        pred_file = os.path.join(dir_name, f'{file_type}_predicts.csv')
        setattr(model, f'{attr_name}_predicts', pd.read_csv(pred_file, index_col=pred_index))
    except:
        pass

    
def load_epoch_files(model, model_dir, file_type, attr_name=None, multi_source=False):
    attr_name = attr_name or file_type
    stats_attr = {}
    preds_attr = {}
    epoch_dirs = glob.glob(os.path.join(model_dir, 'epoch*'))
    epoch_dirs = [epoch for epoch in epoch_dirs if os.path.isdir(epoch)]
    epoch_dirs = sort_file_list(epoch_dirs, 'epoch')
    for epoch_dir in epoch_dirs:
        epoch = os.path.basename(epoch_dir)
        try:
            stats_file = os.path.join(epoch_dir, f'{file_type}_stats.csv')
            stats_attr[epoch] = pd.read_csv(stats_file, index_col=0)
            pred_index = (0,1) if multi_source else 0
            pred_file = os.path.join(epoch_dir, f'{file_type}_predicts.csv')
            preds_attr[epoch] = pd.read_csv(pred_file, index_col=pred_index)
        except:
            pass
    if len(stats_attr) > 0:
        setattr(model, f'epoch_{attr_name}_stats', stats_attr)
    if len(preds_attr) > 0:
        setattr(model, f'epoch_{attr_name}_predicts', preds_attr)

        
def import_model_class(model_class):
    """Imports an LFMC model class
    

    Parameters
    ----------
    model_class : str
        The name of the LFMC model class.

    Returns
    -------
    class
        The imported LFMC model class.

    """
    module = __import__('lfmc_model', fromlist=[model_class])
    return getattr(module, model_class)


def load_model(model_dir, epoch=None):
    model_file = 'model_params.json'
    try:
        with open(os.path.join(model_dir, model_file), 'r') as f:
            model_params = ModelParams(f)
    except:
        raise FileNotFoundError(f'Model parameters missing: {model_file}')
    model_class = model_params.get('modelClass', 'LfmcModel')
    model = import_model_class(model_class)()
    model.load(model_dir)
    multi_source = model_params.get('sourceNames', None)
    if epoch is None:
        dir_name = model_dir
        if model_params['evaluateEpochs']:
            for test_name in ['test', 'val', 'train']:
                load_epoch_files(model, model_dir, test_name, multi_source=multi_source)
                load_epoch_files(model, model_dir, f'fold_{test_name}', multi_source=multi_source)
            
    else:
        dir_name = os.path.join(model_dir, f'epoch{epoch}')
        if os.path.exists(dir_name):
            model.model_dir = dir_name
            model.params['modelDir'] = dir_name
            model.params['epochs'] = epoch
        else:
            dir_name = model_dir

    for test_name in ['test', 'val', 'train']:
        load_files(model, dir_name, test_name, multi_source=multi_source)
        load_files(model, dir_name, f'fold_{test_name}', multi_source=multi_source)
    return model


class LfmcModel():
    """A Keras Model for LFMC estimation
    
    Includes methods to compile, train and evaluate a keras model, and
    to save the models and other outputs. It is intended as an abstract
    class. Subclasses should implement a build method to construct the
    model.
    
    Parameters
    ----------
    params: ModelParams, optional
        All parameters needed for building the Keras model. Default is
        None. The default can be used when creating an object for a
        previously built and trained model. The params will then be
        loaded when the load method is called.
    
    inputs: dict, optional
        If specified, the init method will build and compile the model.
        The parameter should contain a key for each input and the
        values an array of the correct shape. Default is None, meaning
        the model is not built or compiled.
        
    Attributes
    ----------
    params: ModelParams
        All parameters needed for building the Keras model
        
    monitor: str
        Indicates if the callbacks should monitor the training loss
        (``loss``) or validation loss (``val_loss``). Set to ``val_loss``
        if a validation set is used (i.e. ``params['valSize']`` > 0),
        else set to ``loss``.
    
    model_dir: str
        The directory where model outputs will be stored. Identical to
        ``params['modelDir']``
    
    callback_list: list
        A list of keras callbacks. Always includes the ModelCheckpoint
        callback and optionally includes the EarlyStopping callback.
    
    model: keras.Model
        The keras model
        
    derived_models: dict
        Dictionary of models derived from the trained model using the
        ``best_model``, ``merge_models``, or ``ensemble_models``
        methods. Keys are the model names, values are type keras.Model,
        or a list of keras.Models if an ensemble model.
    
    history: list
        The monitored loss and other metrics at each checkpoint.
    """
    
    input_list = []
    model_blocks = {}
    
    @classmethod
    def get_model_blocks(cls):
        return cls.model_blocks
    
    def __init__(self, params=None, inputs=None):
        if params is not None:
            self._setup(params)
        self.derived_models = {}
        self.train_time = 0
        self.build_time = 0
        self.losses = None
        self.loss_weights = None
        if inputs is not None:
            self.build_and_compile(inputs)

    def _setup(self, params):        
        self.params = params
        self.model_dir = params['modelDir']
        self.monitor = 'val_loss' if params['valSize'] else 'loss'
        self._set_random_seeds()
#        tf.debugging.set_log_device_placement(True)
        self._make_temp_dir()
        self._config_gpus()
        if params.get('saveFiles', True):
            self.params.save('model_params.json')

    def _set_random_seeds(self):
        seed = self.params.get('modelSeed', self.params['randomSeed'])
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def _make_temp_dir(self):
        self.temp_dir = None
        while not self.temp_dir:
            sub_dir = ''.join(random.choices(string.ascii_lowercase, k=8))
            temp = os.path.join(self.params['tempDir'], sub_dir)
            try:
                os.makedirs(temp)
                self.temp_dir = temp
            except:
                pass

    def _config_gpus(self):
        self.strategy = None
        if self.params['deterministic']:
            # Currently the only way to ensure a fully deterministic run is to disable GPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:  # only configure GPUs if a non-deterministic run is ok
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            if self.params['mixedPrecision']:
                keras.mixed_precision.set_global_policy(self.params['mixedPrecision'])
            gpu_memory = self.params.get('gpuMemory', 0)
            gpus = tf.config.list_physical_devices('GPU')
            if gpu_memory:
                mem_config = tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory)
                for device in gpus: #gpu_devices:
                    tf.config.set_logical_device_configuration(device, [mem_config])
        
    def _inputs_to_list(self, inputs, build=False):
        ps = 'parent'
        sources = self.params['dataSources'].copy()
        if ps in self.input_list and ps in inputs.keys() and ps not in sources:
            sources.append(ps)
        if build:
            return [keras.Input(inputs[source].shape[1:], name=source)
                    for source in self.input_list]
        else:
            return [inputs[source] for source in sources]
    
    def _targets_to_list(self, targets):
        if targets is None:
            return targets
        elif (targets.ndim == 2) and self.params['multiTarget']:
            if self.params['diagnostics']:
                print('Converting target to list')
            return [x for y, x in targets.items()]
        else:
            if self.params['diagnostics']:
                print(f'Target shape is: {targets.shape}')
            return targets

    def _build_inputs(self, inputs):
        ps = 'parent'
        if self.params['dataSources'] == []:
            self.params['dataSources'] = self.input_list
        elif not set(self.params['dataSources']).issubset(self.input_list):
            raise ValueError(
                f"One or more invalid sources: {self.params['dataSources']}")
        build_data = {}
        for source in self.params['dataSources']:   # self.input_list:
            build_data[source] = keras.Input(inputs[source].shape[1:], name=source)
        if ps in self.input_list and ps in inputs.keys() and ps not in self.params['dataSources']:
            parent_size = inputs[ps].shape[1:]
            build_data[ps] = keras.Input(parent_size, name=ps)
        return build_data

    def _merge_inputs(self, inputs, merge_function):
        temp_inputs = []
        for input_ in inputs:
            if input_ is not None:
                temp_inputs.append(input_)
        if len(temp_inputs) > 1:
            return merge_function(temp_inputs)
        elif len(temp_inputs) == 1:
            return temp_inputs[0]
        else:
            return None
        
    def _eval_param(self, param, starts_with=['keras']):
        """Evaluates a string as code
    
        If ``param`` is prefixed with ``starts_with`` or one of the
        elements in ``starts_with`` (if a list), it is assumed to be
        code and evaluated. Other strings are returned unchanged.
        
        If model parameters are set as keras classes or functions, they
        cannot be converted to JSON and stored. Setting the model
        parameter to a string of the function call allows the full set
        of model parameters to be stored as text, while allowing
        specification of any valid parameter.
        
        Note
        ----
        Currently only implemented for optimizer and kernel_regularizer
        keras parameters (regulariser and optimiser) model parameters.
    
        Parameters
        ----------
        param : str or object
            The string to evaluate. Not evaluated, if not a string.
        starts_with : str or list, optional
            Only evaluate strings prefixed by this string (if str), or
            by one of the strings in the list. The default is ['keras'].
    
        Returns
        -------
        object
            The results of evaluating ``param``. If ``param`` is not a
            string or does not start with ``starts_with``, ``param`` is
            returned unchanged.
        """
        if not isinstance(param, str):
            return param
        if not isinstance(starts_with, list):
            starts_with = [starts_with]
        if any([param.startswith(s) for s in starts_with]):
            return eval(param)
        else:
            return param
    
    def _conv1d_block(self, name, conv_params):
        """Generates layers for a 1d-convolution block
    
        Layers generated are: Conv1D, BatchNormalization, Activation,
        Dropout and AveragePooling1D. Conv1d and Activation layers are
        always generated. Other layers are optional and only included
        if the relevant model_params and conv_params are set as below.
        
        Parameters
        ----------
        name : str
            The name of the block. Used to generate unique names for
            each layer.
        model_params : ModelParams
            Uses the following ModelParams values:
              - initialiser  
              - regulariser  
              - activation  
              - dropoutRate (Dropout layer included if > 0)
        conv_params : dict
            The convolutional parameters for this block. Required keys:
              - filters
              - kernel
              - stride
              - bnorm (BatchNormalization layer included if True)
              - poolSize (AveragePooling1D layer included if > 0)
    
        Returns
        -------
        block : list of keras layers
            The keras layers that comprise the convolution block.
        """
        block = []
        block.append(keras.layers.Conv1D(
                name=name + '_conv1d',
                filters=conv_params['filters'],
                kernel_size=conv_params['kernel'],
                strides=conv_params['stride'],
                dilation_rate=conv_params['dilation'],
                padding=self.params.get('convPadding', "same"),
                kernel_initializer=self.params['initialiser'],
                kernel_regularizer=self._eval_param(self.params['regulariser'])))
        if conv_params['bnorm']:
            block.append(keras.layers.BatchNormalization(name=name + '_bnorm', axis=-1))
        block.append(keras.layers.Activation(self.params['activation'], name=name + '_act'))
        if self.params['dropoutRate'] > 0:
            block.append(keras.layers.Dropout(self.params['dropoutRate'], name=name + '_dropout'))
        if conv_params['poolSize'] > 0:
            block.append(keras.layers.AveragePooling1D(
                pool_size=conv_params['poolSize'],
                padding=self.params.get('poolPadding', "valid"),
                name=name + '_pool'))
        return block
    
    def _dense_block(self, name, fc_params):
        """Generates layers for a fully-connected (dense) block
     
        Layers generated are: Dense, BatchNormalization, Activation,
        and Dropout. Dense and Activation layers are always generated.
        The other layers are optional and only included if the relevant
        model_params and conv_params are set as below.
    
        Parameters
        ----------
        name : str
            The name of the block. Used to generate unique names for each
            layer.
        model_params : ModelParams
            Uses the following ModelParams values:
                initialiser
                regulariser
                activation
                dropoutRate (Dropout layer included if > 0)            
        fc_params : dict
            The fully-connected parameters for this block. Required keys
            are:
                units
                bnorm (BatchNormalization layer included if True)
    
        Returns
        -------
        block : list of keras layers
            The keras layers that comprise the fully-connected block.
        """
        block = []
        block.append(keras.layers.Dense(
                fc_params['units'],
                name=name + '_dense',
                kernel_initializer=self.params['initialiser'],
                kernel_regularizer=self._eval_param(self.params['regulariser'])))
        if fc_params['bnorm']:
            block.append(keras.layers.BatchNormalization(name=name + '_bnorm', axis=-1))
        block.append(keras.layers.Activation(self.params['activation'], name=name + '_act'))
        if self.params['dropoutRate'] > 0:
            block.append(keras.layers.Dropout(self.params['dropoutRate'], name=name + '_dropout'))
        return block

    def _add_block(self, block_name, block_func, input_):
        if input_ is None:
            return None
        x = input_
        short_name = block_name.replace('Conv', '')
        for i, block_params in enumerate(self.params['blocks'].get(block_name, [])):
            for layer in block_func(f'{short_name}{i}', block_params):
                x = layer(x)
        return x

    def _final_layer(self, block_name='final', input_=None):
        if input_ is None:
            return None
        x = input_
        if self.params.get('classify'):
            units = self.params.get('numClasses', 2)
            if units <= 2:
                units = 1
                act = 'sigmoid'
            else:
                act = 'softmax'
        else:
            units = 1
            act = 'linear'
        x = keras.layers.Dense(units, name=block_name, dtype='float32',
                               kernel_initializer=self.params['initialiser'],
                               activation=act)(x)
        return x
    
    def get(self):
        return self
    
    def ready(self):
        return True
    
    def build(self, inputs):
        raise NotImplementedError

    def compile(self, steps_per_execution=None):
        """Compiles the model
        
        Compiles the model using the optimizer, loss function and
        metrics from params.

        Returns
        -------
        None.
        """
        metrics = [self._eval_param(metric, ['keras', 'custom'])
                   for metric in self.params['metrics']]
        if self.losses is None:
            self.losses = self._eval_param(self.params['loss'], ['keras', 'custom'])
        steps_per_execution = steps_per_execution or self.params['stepsPerExec']
        self.model.compile(optimizer=self._eval_param(self.params['optimiser']),
                           loss=self.losses,
                           loss_weights=self.loss_weights,
                           metrics=metrics,
                           steps_per_execution=steps_per_execution,
                           jit_compile=self.params['enableXla'])

    def build_and_compile(self, inputs, compile_only=False, load_weights=False):
        start_build_time = time.time()
        if self.params is None:
            raise ValueError('Model parameters must be specified to build model')
        if not compile_only:
            self.build(inputs)
        if load_weights:
            pretrained_model = os.path.join(self.params['pretrainedModel'], 'base.h5')
            self.model.load_weights(pretrained_model, by_name=True)
        self.compile()
        self.next_epoch = 0
        self.set_callbacks()
        self.build_time = round(time.time() - start_build_time, 2)

    def transfer(self, *args, **kwargs):
        xfer = self.params['transferModel']
        if xfer:
            if xfer.get('preTransfer', False):
                import_transfer_method(xfer['preTransfer'])(self, *args, pre_transfer=True, **kwargs)
            if xfer.get('method', None):
                import_transfer_method(xfer['method'])(self, *args, **kwargs)

    def set_callbacks(self):
        """Creates a list of the model callbacks.
        
        The callback list contains a ModelCheckpoint callback, so a
        checkpoint is taken after each epoch. If earlyStopping is set
        in params, an EarlyStopping checkpoint is also created. The
        callbacks are saved to the callback_list attribute.

        Returns
        -------
        None.
        """
        if self.params['checkpoint']:
            checkpoint_path = os.path.join(self.temp_dir, '_{epoch:04d}_temp.h5')
            checkpoint = keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor=self.monitor,
                    verbose=self.params['verbose'],
                    save_best_only=False,
                    save_weights_only=False,
                    mode='min',
                    save_freq='epoch')
            self.callback_list = [checkpoint]
        else:
            self.callback_list = []
        if self.params['earlyStopping'] > 1:
            early_stop = keras.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    min_delta=0,
                    patience=self.params['earlyStopping'],
                    verbose=self.params['verbose'],
                    mode='auto')
            self.callback_list.append(early_stop)
        if self.params['tensorBoard']:
            if isinstance(self.params['tensorBoard'], dict):
                kwargs = self.params['tensorBoard']
            else:
                kwargs = None
            tensor_board = keras.callbacks.TensorBoard(
                log_dir=self.model_dir, histogram_freq=1, **kwargs)
            self.callback_list.append(tensor_board)
            

    def train(self, Xtrain, ytrain, Xval=None, yval=None, weights=None):
        """Trains the model
        
        Parameters
        ----------
        Xtrain : dict
            The training features. The dict keys are the sources and
            the values are arrays of the data for each source.
        ytrain : array
            The training labels.
        Xval : dict, optional
            The validation features in the same format as Xtrain. The
            default is None. Ignored if model params has valSize = 0.
        yval : TYPE, optional
            The validation labels. The default is None. Ignored if the
            model params has valSize = 0.

        Returns
        -------
        dict
            Dictionary of the training results, with keys:
            'minLoss' - the minimum loss from the training checkpoints
            'history' - the loss and metrics at all checkpoints
            'runTime' - the training time in seconds
        """
        if self.params['diagnostics']:
            for i, x in Xtrain.items():
                print(f'{i} input shape: {x.shape}')
                print(f'{i} data: {x[10].round(3)}')
            print(f'Target shape: {ytrain.shape}')
            print(f'Target data: {ytrain[:100].round(2)}')
            print('Batch size: ', self.params['batchSize'])
            print('Epochs: ', self.next_epoch, self.last_epoch)

        Xtrain = self._inputs_to_list(Xtrain)
        Xval = self._inputs_to_list(Xval)
        ytrain = self._targets_to_list(ytrain)
        yval = self._targets_to_list(yval)
        
        start_train_time = time.time()
        if self.params['valSize']:
            hist = self.model.fit(x=Xtrain, y=ytrain, sample_weight=weights,
                    epochs=self.last_epoch, initial_epoch = self.next_epoch,
                    batch_size=self.params['batchSize'], shuffle=self.params['shuffle'],
                    verbose=self.params['verbose'], callbacks=self.callback_list,
                    validation_data=(Xval, yval))
        else:
            hist = self.model.fit(x=Xtrain, y=ytrain, sample_weight=weights,
                    epochs=self.last_epoch, initial_epoch = self.next_epoch,
                    batch_size=self.params['batchSize'], shuffle=self.params['shuffle'],
                    verbose=self.params['verbose'], callbacks=self.callback_list)
        self.train_time += round(time.time() - start_train_time, 2)
        if self.next_epoch:
            self.history = pd.concat([self.history, pd.DataFrame(hist.history)], ignore_index=True)
        else:
            self.history = pd.DataFrame(hist.history)
        self.history.set_index(self.history.index + 1, inplace=True)
        self.next_epoch = self.last_epoch
        if self.params['saveTrain'] and (self.last_epoch == self.params['epochs']):
            self.save_hist()
        return {'minLoss': np.min(hist.history[self.monitor]),
                'history': hist.history,
                'runTime': self.train_time}

    def predict(self, X, model_name='base', ensemble=np.mean, batch_size=1024, verbose=0,
                get_aux=False):
        """Predicts labels from a model
        
        Parameters
        ----------
        X : dict
            The features to predict from.
        model_name : str, optional
            The name of the model to use for prediction. The 'base'
            model (i.e. fully-trained model) is used if no model
            specified. The default is 'base' (the fully trained model).
        ensemble : function, optional
            If the model is an ensemble, the function to use to combine
            the individual predictions. The default is np.mean.
        batch_size : int, optional
            keras.Model.predict batch_size parameter. The default is 1024.
        verbose : int, optional
            keras.Model.predict verbose parameter. The default is 0.

        Returns
        -------
        yhat : array
            The predicted labels.
        """
        X = self._inputs_to_list(X)
        model = self.model if model_name == 'base' else self.derived_models[model_name]
        if type(model) is list:
            yhat = [m.predict(X, batch_size=batch_size, verbose=verbose) for m in model]
            yhat = ensemble(np.hstack(yhat), axis=1)
        else:
            yhat = model.predict(X, batch_size=batch_size, verbose=verbose)
            if isinstance(yhat, tuple):
                aux = yhat[1:]
                yhat = yhat[0]
            else:
                aux = []
            if self.params['classify'] and (self.params['numClasses'] > 2):
                yhat = yhat.argmax(axis=1)
            else:
                yhat = yhat.flatten()
            if verbose:
                print(yhat)
                for n_, y_ in enumerate(aux):
                    print(f'Auxiliary predictions {n_}: {aux}')
        if get_aux:
            return yhat, aux
        else:
            return yhat

    def evaluate(self, X, y, model_name='base', ensemble=np.mean, get_stats=True,
                 plot=True, fig_name=None):
        """Evaluates the model
        
        Evaluate the model using X as the test data and y as the labels
        for the test data.

        Parameters
        ----------
        X : dict
            The test data.
        y : array
            The test labels.
        model_name : str, optional
            Name of the model to use for the predictions. The default
            is 'base' (the fully trained model).
        ensemble : function, optional
            The function to use if the model is an ensemble. The
            default is np.mean.
        get_stats : bool, optional
            Flag indicating if prediction statistics should be
            calculated. The default is True.
        plot : bool, optional
            Flag indicating if a scatter plot of the results should be
            created. The default is True.
        fig_name : str, optional
            Name for the plot figure. If none, the model name is used.
            The default is None.

        Returns
        -------
        dict
            Dictionary of the evaluation results, with keys:
              - 'predict' - the predicted values
              - 'stats' - the evaluation statistics (bias, R, R2, RMSE,
                ubRMSE)
              - 'runTime' - the prediction time in seconds
        """
        start_test_time = time.time()
        if len(y) == 0:
            yhat = np.array([])
            aux = []
        else:
            yhat, aux = self.predict(X, model_name=model_name, ensemble=ensemble,
                                     verbose=self.params['verbose'], get_aux=True)
        test_time = round(time.time() - start_test_time, 2)
        if get_stats:
            if (y.ndim == 2) and self.params['multiTarget']:
                y = self._targets_to_list(y)
                y = y[0]
                yhat = yhat[0]
            stats = calc_statistics(y, yhat, classify=self.params['classify'])
        else:
            stats = None
        if plot:
            if fig_name is None:
                fig_name = 'base' if model_name is None else model_name
            fig = plot_results(f'{fig_name} Results', y, yhat, stats)
            fig.savefig(os.path.join(self.model_dir, fig_name + '.png'), dpi=300)
        return {'predict': yhat, 'auxPredicts': aux, 'stats': stats, 'runTime': test_time}

    def summary(self):
        """Prints the model summary.
        
        Returns
        -------
        None.
        """
        self.model.summary()
        
    def weight_counts(self):
        """Gets the number of model weights
        
        Calculates and returns the number of trainable and
        non-trainable weights in the model.

        Returns
        -------
        trainable : np.int32
            The number of trainable weights in the model.
        untrainable : np.int32
            The number of non-trainable weights in the model.

        """
        trainable = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        untrainable = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        return (trainable, untrainable)
    
    def plot(self, dir_name=None, file_name='model_plot.png'):
        """Saves model plot
        
        Calls the Keras plot_model utility to create an image of the
        model network.

        Parameters
        ----------
        file_name : str
            Name of the plot file (relative to self.model_dir).

        Returns
        -------
        None.
        """
        outdir = dir_name or self.model_dir
        out_file = os.path.join(outdir, file_name)
        keras.utils.plot_model(self.model, to_file=out_file, show_shapes=True,
                               show_layer_names=True)

    def load(self, model_dir):
        """Loads the model
        
        Loads a saved model from disk. Run this method after creating
        the instance with no parameters.

        Parameters
        ----------
        model_dir : str
            The full path name of the directory storing the model.

        Returns
        -------
        None.
        """
        self.model_dir = model_dir # os.path.join(model_dir, '')  # add separator if necessary
        with open(os.path.join(model_dir, 'model_params.json'), 'r') as f:
            self.params = ModelParams(source=f)
        self.derived_models = {}
        self.train_time = 0
        self.build_time = 0
        self.monitor = 'val_loss' if self.params['valSize'] else 'loss'
        try:
            self.history = pd.read_csv(os.path.join(model_dir, 'train_history.csv'))
            self.history.set_index(self.history.index + 1, inplace=True)
        except:
            self.history = pd.DataFrame()
            
    def setup_finetune(self, params, inputs=None, target_size=0, source_size=0):
        self._setup(params)
        self.build_and_compile(inputs=inputs, load_weights=True)
        if params.get('transferModel'):
            self.transfer(target_size, source_size)

    def save_hist(self):
        """Saves the model history
        
        Saves the model plot, model parameters and training history
        Saves the model training history

        Returns
        -------
        None.
        """
        self.history.to_csv(os.path.join(self.model_dir, 'train_history.csv'), index=False)
        
    def load_model(self, model_name='base'):
        """Loads a derived model
        
        Parameters
        ----------
        model_name : str
            The name of the derived model. This is assumed to be the
            filename (relative to the model directory), excluding the
            suffix, for a single model; or a directory containing the
            individual files for an ensemble.

        Returns
        -------
        None.
        """
        try:
            saved_model = keras.models.load_model(os.path.join(self.model_dir, f'{model_name}.h5'))
        except:
            model_files = glob.glob(os.path.join(self.model_dir, model_name, "*.h5"))
            saved_model = [keras.models.load_model(m) for m in model_files]
        if model_name == 'base':
            self.model = saved_model
        else:
            self.derived_models[model_name] = saved_model
        
    def save_to_disk(self, model_name=None, model_list=[], prefix=''):
        """Saves models to disk
        
        Saves the specified model or models to disk. If neither
        ``model_name`` or ``model_list`` are specified, all models are
        saved.

        Parameters
        ----------
        model_name : str, optional
            Name of the model to save. The default is None.
        model_list : list, optional
            List of models to save. The default is [].

        Returns
        -------
        None.
        """
        if not(model_name or model_list):
            model_list = ['base'] + list(self.derived_models.keys())
        for mn in model_list or [model_name]:
            save_name = '_'.join([prefix, mn]) if prefix else mn
            if mn == 'base':
                self.model.save(os.path.join(self.model_dir, f'{save_name}.h5'))
            else:
                model_ = self.derived_models[mn]
                if type(model_) is list:
                    save_dir = os.path.join(self.model_dir, save_name)
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                    os.makedirs(save_dir)
                    for i, m in enumerate(model_):
                        m.save(os.path.join(save_dir, f"m{i:03d}.h5"))
                else:
                    model_.save(os.path.join(self.model_dir, f'{save_name}.h5'))
        
    def _save_model(self, new_model, model_name):
        self.derived_models[model_name] = new_model
        
    def _get_model_list(self, sort=True):
        model_list = glob.glob(os.path.join(self.temp_dir, '*_temp.h5'), recursive=True)
        if sort:
            model_list.sort()
        return model_list
    
    def clear_model(self):
        """Clears the model
        
        Removes as much of the model as possible, including:
          - model and derivedModel attributes
          - keras backend session
          - deallocate GPU and memory
          - runs garbage collection
          - delete all checkpoint files
            
        After running clear_model, the model instance still exists but
        no keras components will exist. As the checkpoint files are
        removed, no new derived models can be created. All other
        components are saved to disk and can be reloaded using the load
        and load_model methods.

        Returns
        -------
        None.
        """
        self.model = None
        self.derived_models = None
        self.callback_list = None
        keras.backend.clear_session()
        gc.collect()
        if hasattr(self, 'temp_dir'):
            for file in self._get_model_list(sort=False):
                os.remove(file)
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
        
    def get_models(self, models=None, last_n=False, load=False):
        """Gets a subset of checkpoint models
        

        Parameters
        ----------
        models : int or list of int, optional
          - If a list of int, then interpreted as a list of checkpoint
            numbers.
          - If an int and ``last_n`` is True, then interpreted as
            requesting the last N models, where N is ``abs(models)``.
          - If an int and ``last_n`` is False, then interpreted as the
            required checkpoint number, or if negative, the Nth last
            checkpoint.
          - If None, all checkpoints are returned. The default is None.
        last_n : bool, optional
            If True and ``models`` is type int, interpret models as a
            request for the last N models. Ignored if ``models`` is
            type str. The default is False.
        load : bool, optional
            If True, load the set of models from the saved checkpoints
            and return a list of the models. If False, return a list of
            model file names. The default is False.

        Returns
        -------
        str, keras.Model, or list
            If a single model requested then either the filename of the
            checkpoint (``load=False``) or the requested checkpoint as
            a keras Model (``load=True``). If more than one model
            requested, then the checkpoint filenames (``load=False``)
            or the checkpoint models (``load=True``) as a list.
        """
        model_list = self._get_model_list()
        model_series = pd.Series(model_list, index=self.history.index)
        if models is None:   # Return the entire series of models
            return model_series
        # Select the requested model(s)
        if type(models) is int and last_n:
            modelx = model_series[-abs(models):]
        else:
            if type(models) is int and models < 0:
                modelx = model_list[models]   # Use the list as np.series[-n] fails
            else:
                modelx = model_series[models]
        if load:    # Load the requested model(s)
            if type(modelx) is str:  # Only one model requested
                return keras.models.load_model(modelx)
            else:
                return [keras.models.load_model(m) for m in modelx]
        else:
            return modelx

    def merge_models(self, model_name, models=None):
        """Creates a model by merging a set of models
        
        Create a model from a set of model checkpoints by averaging the
        equivalent weights from each model. A merged model is similar
        to an ensemble but usually slightly less accurate than an
        ensemble of the same set. The advantage is that it is much
        faster to predict from the merged model.

        Parameters
        ----------
        model_name : str
            A name for the merged model.
        models : list, optional
            The models to merge. See ``get_models model`` parameter for
            details of how this parameter is used. The default is None,
            meaning all checkpoints are merged.

        Returns
        -------
        None.
        """
        weights = [m.get_weights() for m in self.get_models(models, last_n=True, load=True)]
        # average weights
        new_weights = [np.array(w).mean(axis=0) for w in zip(*weights)]
        model_conf = self.model.get_config()
        new_model = keras.models.Model.from_config(model_conf)
        new_model.set_weights(new_weights)
        self._save_model(new_model, model_name)

    def ensemble_models(self, model_name, models=None):
        """Creates a model by ensembling a set of models
        
        Create a model as an ensemble of checkpoints.

        Parameters
        ----------
        model_name : str
            A name for the ensemble model.
        models : list, optional
            The models to ensemble. See ``get_models model`` parameter
            for details of how this parameter is used. The default is
            None, meaning all checkpoints are ensembled.

        Returns
        -------
        None.
        """
        ensemble = self.get_models(models, last_n=True, load=True)
        self._save_model(ensemble, model_name)

    def best_model(self, model_name=None, n=1, merge=True):
        """Creates a model using the best ``n`` checkpoints
        
        Create a model using the checkpoint(s) with the lowest loss.

        Parameters
        ----------
        model_name : str, optional
            A name for the model. If ``None``, a name is generated from
            the other parameters.
        n : int
            The number of checkpoints to use. If 1, then a model is
            created using a single checkpoint with the lowest loss. If
            > 1, create a model using the ``n`` checkpoints with the
            lowest loss. The default is 1.
        merge : bool, optional
            If n > 1, indicates whether to merge (True) or ensemble
            (False) the best ``n`` checkpoints. The default is True.

        Returns
        -------
        None.
        """
        best = list(self.history[self.monitor].nsmallest(n).index)
        if len(best) == 1:
            best = best[0]
            best_model = self.get_models(best, load=True)
            self._save_model(best_model, model_name or 'best')
        elif merge:
            self.merge_models(model_name or f'merge_best{n:02d}', best)
        else:
            self.ensemble_models(model_name or f'ensemble_best{n:02d}', best)
        return best
    
    def plot_train_hist(self, file_name=None, metric=None, rolling=10):
        """Creates a plot of the training result at each epoch.

        Parameters
        ----------
        file_name : str, optional
            The file name for the plot, relative to the model directory.
            The `.png` extension is appended to the file name. If
            ``None`` the name ``training_{metric}.png`` is used. The
            default is None.
        metric : str, optional
            The name of the metric to plot. If ``None``, the checkpoint
            monitored metric is used, otherwise can be any metric
            included in the ``metrics`` model parameter. The default is
            None.
        rolling : int, optional
            The number of epochs to use to plot the rolling (moving)
            average. If 1 (or less), no rolling average is plotted. The
            default is 10.

        Returns
        -------
        None.
        """
        if metric is None:
            metric = self.monitor
        if file_name is None:
            file_name = f'training_{metric}'
        plt.figure(figsize=(5, 5))
        plt.plot(self.history[metric], label='Epoch values')
        if rolling > 1:
            plt.plot(self.history[metric].rolling(rolling, center=True).mean(),
                     label=f'Moving average({rolling})')
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        min_y = round(self.history[metric].min()*2-50, -2)/2
        if self.params['epochs'] >= 10:
            max_y = self.history[metric].iloc[5:].max()
        else:
            max_y = self.history[metric].max()
        if max_y > 100:
            max_y = round(max_y * 2 + 50, -2) / 2
        elif max_y > 10:
            max_y = round(max_y * 2 + 5, -1) / 2
        else:
            max_y = round(max_y + 1, 0)
        plt.axis([0, self.params['epochs'], min_y, max_y])
        plt.legend()
        plt.title(f'Training Results - {metric}')
        plt.savefig(os.path.join(self.model_dir, file_name + '.png'), dpi=300)
        plt.close()


class LfmcTempCnn(LfmcModel):
    """A TempCNN for LFMC estimation
    
    A subclass of LfmcModel that builds a TempCNN. It caters for these
    inputs:
      - optical - a daily timeseries of optical reflectance data
      - weather - a daily timeseries of weather data
      - aux - the auxiliary data
        
    blocks:
      - opticalConv - convolves the optical data
      - weatherConv - convolves the weather data
      - fc - the fully-connected or dense layers
    """
    
    input_list = ['optical', 'weather', 'aux', 'parent']
    model_blocks = {'opticalConv': 'conv1d',
                    'weatherConv': 'conv1d',
                    'fc': 'dense' }
    
    def build(self, inputs):
        """Builds the model.
        
        Build the model with layers and hyper-parameters as specified
        in params. The shape of the inputs are used to define the keras
        input layers. 
        
        Parameters
        ----------
        inputs : dict
            Should contain a key for each input and values an array of
            the correct shape. The dimensions of the array are used to
            build the model layers.

        Returns
        -------
        None.
        """
        flatten = keras.layers.Flatten
        concatenate = keras.layers.Concatenate
        inputs = self._build_inputs(inputs)
        # Convolve optical data if required
        optical = inputs.get('optical')
        optical = self._add_block('opticalConv', self._conv1d_block, optical)
        # Convolve weather data if required
        weather = inputs.get('weather')
        weather = self._add_block('weatherConv', self._conv1d_block, weather)
        # Stack optical and weather, and convolve to same shape as soilmoist data
        optical = flatten(name='optical_flatten')(optical) if optical is not None else None
        weather = flatten(name='weather_flatten')(weather) if weather is not None else None
        daily = self._merge_inputs([optical, weather], concatenate(name='daily'))
        # Combine EO and auxiliary features
        aux = inputs.get('aux')
        parent = inputs.get('parent')
        full = self._merge_inputs([daily, aux, parent], concatenate(name='concat'))
        # Add the dense layers
        full = self._add_block('fc', self._dense_block, full)
        # Add the output layer
        full = self._final_layer('final', full)
        self.model = keras.Model(inputs=inputs.values(), outputs=full,
                                 name=self.params['modelName'])
        if self.params['diagnostics']:
            self.summary()

