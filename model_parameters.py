"""The Model Parameters dictionary"""

import json
import os
import pprint

from copy import deepcopy

class ParamDict(dict):
    """A parameter dictionary
    
    Extends the dictionary class by adding a help function and a
    default set of keys. Includes a method to save the parameters to a
    json fileor as a json string. The __init__ function allows loading
    parameters from a json string or file as well as from a dict.
    
    Parameters
    ----------
    source : None, dict, str, or file object, optional
      - If dict: A dictionary containing all parameters
      - If str: A string representation of parameters in JSON format.
      - If file object: An open JSON file containing all parameters.
      - If None: The object is initialised with defaults.
      - The default is None.
        
    Attributes
    ----------
    _param_help: dict
        A dictionary containing the ``help`` text for each model
        parameter. The ``general`` key contains the ``help`` text for
        the object.
    """

    def __init__(self, source=None):
        parameters = self.set_defaults()
        if isinstance(source, dict):   # Set parameters from a dictionary
            parameters.update(source)
        elif type(source) is str:  # Set parameters from a JSON string
            parameters.update(json.loads(source))
        elif source:               # Set parameters from a JSON input file
            parameters.update(json.load(source))
        super(ParamDict, self).__init__(parameters)
        
    def set_defaults(self):
        """Returns the default parameter dictionary.
        
        Returns
        -------
        dict
            An empty dictionary.
        """
        return {}

    def __str__(self):
        return pprint.pformat(self, width=100, sort_dicts=False)

    def save(self, file_stream=None, directory=None):
        """Saves the model parameters
        
        Convert the model parameters dictionary to a JSON string and
        optionally save to a file.
        
        Parameters
        ----------
        file_stream : str or file handle, optional
            Either the name of the output file, or a file handle for
            the output file. If None, the converted JSON string is
            returned. The default is None.
        directory : str, optional
            If file_stream is str, the directory the output file is
            created in. It will be created if it doesn't exist. The
            default is ``self['modelDir']`` if this key exists, else
            None.

        Returns
        -------
        str
            The JSON representation of the model parameters. Only
            returned if no file stream parameter specified.

        """
        if file_stream is None:               # No output file, return parameters as a JSON string
            return json.dumps(self, indent=2)
        elif isinstance(file_stream, str):    # File name provided, open file and save parameters
            if not directory:
                try:
                    directory = self['modelDir']
                except:
                    raise FileNotFoundError('No directory specified')
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(os.path.join(directory, file_stream), 'w') as f:
                json.dump(self, f, indent=2)
        else:                                 # File stream provided, save parameters
            json.dump(self, file_stream, indent=2)

    def help(self, key=None):
        """Prints a help message
        
        Prints the general help message if no key provided, or the help
        message for the specified key.
        
        Parameters
        ----------
        key : str, optional
            The key for which help is requested. If None, the general
            help message is displayed. The default is None.

        Returns
        -------
        None.
        """
        def pp(text, indent=0, quote=False):
            spaces = " " * indent
            if quote:
                out = pprint.pformat(text).replace("('", "'").replace("')", "'").replace(
                        "\n ", f"\n{spaces}").replace("\\n", "\n")
            else:
                out = pprint.pformat(text).replace("('", "").replace("')", "").replace(
                        "'", "").replace("\n ", f"\n{spaces}").replace("\\n\n", "\n").replace(
                        "\\n", "\n")
            return out

        sep = '\n  '
        if key is None:
            text = pp(self._param_help['general']) + sep + sep.join(self)
        else:
            keyValue = pp(self.get(key, 'not defined'), indent=10, quote=True)
            keyHelp = pp(self._param_help.get(key, 'not available'), indent=8)
            text = f'{key}:\n  value: {keyValue}\n  help: {keyHelp}'
        print(text)

    _param_help = {
        'general': 'Dictionary of parameters. Subclasses "ModelParams" and "ExperimentParams" '
                   'are used to define the required parameters and default settings. For more '
                   'help run ModelParams().help() or ExperimentParams().help().',
    }


class ModelParams(ParamDict):
    """A dictionary for LFMC model parameters
    
    A ParamDict sub-class that creates a dictionary with keys for all
    parameters needed to build a model for LFMC estimation. Defaults
    are set for the parameters if no source is provided or it has
    missing keys.
    
    Parameters
    ----------
        source : None, dict, str, or file object, optional
          - If dict: A dictionary containing all model parameters
          - If str: A string representation of the model parameters in
            JSON format.
          - If file object: An open JSON file containing all model
            parameters.
          - If None: The object is initialised with defaults for all
            model parameters.
          - The default is None.
        model_name : str, optional
            The name of the model. It should be a valid Keras model
            name. The default is 'default_model'.
        * : int or list, optional
            All other parameters are passed to the ``set_layers`` method
            to add the convolutional and fully-connected layers.
        
    Attributes
    ----------
    _param_help: dict
        A dictionary containing the ``help`` text for each model
        parameter. The ``general`` key contains the ``help`` text for
        the object.
    """

    def __init__(self, source=None, model_name='default_model', **blocks):
        super(ModelParams, self).__init__(source)
        if source is None:
            self['modelName'] = model_name
            self.set_layers(**blocks)
        
    def set_defaults(self):
        """Returns the default model parameters dictionary
        
        Returns
        -------
        model_params : dict
            The dictionary of model parameters.
        """
        model_params = {
            'modelName': 'default_model',
            'description': '',
            'modelClass': 'LfmcModel',
            'modelDir': '',
            'tempDir': '',
            'diagnostics': False,
            'dataSources': [],
            'restartRun': None,
            'derivedModels': None,
            'saveModels': False,
            'saveTrain': None,
            'saveValidation': True,
            'plotModel': True,
            'multiSamples': None,
            'deduplicate': False,

            'randomSeed': 1234,
            'modelSeed': 1234,
            'modelRuns': 1,
            'resplit': False,
            'seedList': [],
            
            # Multiprocessing parameters
            'maxWorkers': 1,
            'deterministic': False,
            'gpuDevice': 0,
            'gpuList': [],
            'gpuMemory': 0,

            # Input data parameters
            'inputs': {},
            'samplesFile': None,
            'samplesFilter': None,
            'auxColumns': [],
            'auxAugment': True,
            'auxOneHotCols': [],
            'targetColumn': None,
            'targetTransform': None,
            'targetNormalise': None,
            'classify': False,
            'numClasses': 0,

            # Data splitting parameters
            'splitMethod': None,
            'splitSizes': (0.33, 0.067),
            'splitColumn': None,
            'splitStratify': None,
            'splitYear': None,
            'splitFolds': 0,
            'testFolds': 1,
            'yearColumn': None,
            'yearFolds': None,
            'testAllYears': False,
            'trainAdjust': 0,
            'testAdjust': 0,
            
            # Model chaining parameters
            'parentModel': None,
            'parentFilter': None,
            'parentResult': False,
            
            # Keras common parameters
            'convPadding': 'valid',
            'poolPadding': 'valid',

            # Overfitting controls
            'batchNormalise': False,
            'dropoutRate': 0,
            'regulariser': None,
            'validationSet': False,
            'earlyStopping': False,

            # Fitting parameters
            'epochs': 1,
            'evaluateEpochs': None,
            'batchSize': 32,
            'shuffle': True,
            'verbose': 0,

            # Keras methods
            'optimiser': 'adam',
            'activation': None,
            'initialiser': 'he_normal',
            'loss': None,
            'metrics': None,
        }
        return model_params

    def set_layers(self, **kwargs):
        """Sets model layer parameters
        
        Sets the parameters for the convolutional (``conv``, ``*Conv``)
        and fully connected (``fc``) layers.

        Parameters
        ----------
        conv_layers : int or list, optional
            The number of convolutional layers, or a list where the
            list length is the number of layers and each entry in the
            list is a dictionary of parameters for the layer. If ``0``
            or ``[]``, any existing fc layers are removed. If ``None``,
            any existing layers are not changed. The default is None.
        fc_layers : int or list, optional
            The number of fully connected (or dense) layers, or a list
            where the list length is the number of layers and each
            entry in the list is a dictionary of parameters for the
            layer. If ``0`` or ``[]``, any existing fc layers are
            removed. If ``None``, any existing layers are not changed.
            The default is None.
        * : int or list, optional
            Any other parameters are assumed to be additional
            convolutional layers with the name ``xxxConv``, where
            ``xxx`` is the parameter name truncated at the first ``_``.
            (e.g. to add the modisConv layers, specify modis_layers=n).
            The value is the number of layers, or a list where the list
            length is the number of layers and each entry in the list
            is a dictionary of parameters for the layer. If ``0`` or
            ``[]``, any existing fc layers are removed. If ``None``,
            any existing layers are not changed. The default is None.

        Returns
        -------
        None.
        """
        def block_parms(key):
            if key == 'fc_layers':
                return ['fc', 'Dense']
            elif key == 'conv_layers':
                return ['conv', 'Conv']
            else:
                return [key.split('_')[0] + 'Conv', 'Conv']    

        for key, block_params in kwargs.items():
            block_name, block_type = block_parms(key)
            if isinstance(block_params, int):
                self.add_block(block_name, block_type, block_params)
            else:
                self.add_block(block_name, block_type, len(block_params), block_params)

    def add_block(self, block_name, block_type, num_layers, block_params={}):
        """Sets parameters for a model block
        
        Creates a set of model parameters that will be used to build a
        block of layers. Currently supports (1D) convolutional blocks
        and dense (or fully-connected) blocks. Parameters for each
        layer are set to the defaults for the block type and then
        updated using ``block_params``, if this parameter is set.

        Parameters
        ----------
        block_name : str
            A name for the block.
        block_type : str
            Either "Conv" for a convolutional block, or "Dense" for a
            dense block.
        num_layers : int
            The number layers in the block.
        block_params : dict, optional
            A dictionary of parameters that override or add to the
            default parameter settings.

        Returns
        -------
        None.
        """
        conv_params = {
            'filters': 8,    # Convolution filters
            'kernel': 5,     # Convolution kernel size
            'stride': 1,     # Convolution stride
            'dilation': 1,   # Convolution dilation
            'bnorm': self['batchNormalise'],
            'poolSize': 0,   # 0 to disable pooling
        }

        dense_params = {
            'units': 512,
            'bnorm': self['batchNormalise'],
        }

        # Add the block and set parameters to defaults
        if block_type == 'Conv':
            self[block_name] = [conv_params.copy() for _ in range(num_layers)]
        elif block_type == 'Dense':
            self[block_name] = [dense_params.copy() for _ in range(num_layers)]
        else:
            raise ValueError
        # Update block with supplied parameters
        if block_params:
            for idx, layer in enumerate(self[block_name]):
                layer.update(block_params[idx])
    
    def add_input(self, name, input_params, data_type='ts'):
        """ Adds an input to the model parameters.
        

        Parameters
        ----------
        name : str
            Input name.
        input_params : dict
            Dictionary of parameters. Valid keys are:
            'filename': Required.
                Full path name of the file containing the data for a
                list of file names.
            'channels': Required for time series inputs.
                Number of channels in the dataset.
            'includeChannels': Optional for time series inputs.
                A list of channels to include in the prepared data. By
                default, all channels are included.
            'normalise': Optional. 
                A dictionary containing the method to use to normalise
                the data, plus any parameters required by this method.
            'start': Optional.
                Time series start. The offset from the start of the
                input time series.
            'end': Optional.
                Time series end. The offset from the end of the input
                time series. "None" means no end offset.
        data_type : str, optional
            Input type. Specify 'ts' (the default) if the input is a
            time series. This ensures a default value for all the time
            series input parameters (channels/start/end) is set. If any
            other value is specified, no defaults for these parameters
            are set, although they may be provided in the input_params.

        Returns
        -------
        None.

        """
        all_defaults = {'filename': None, 'normalise': None,}
        ts_defaults = {
            'filename': None,
            'channels': None,
            'includeChannels': [],
            'normalise': {'method': 'minMax', 'percentiles': 2},
            'start': None,
            'end': None,
            }
        self['inputs'][name] = deepcopy(all_defaults)
        if data_type == 'ts':
            self['inputs'][name].update(deepcopy(ts_defaults))
        self['inputs'][name].update(input_params)


    _param_help = {
        'general':        'Dictionary of all parameters used to build an LFMC model. For more '
                          'help run ModelParams().help("parameter").\nAvailable parameters are:',
        'modelName':      'A name for the model; must be a valid Keras model name',
        'description':    'A free-format description of the model - only used for documentation',
        'modelClass':     'The Class for the model. This should be a sub-class of LfmcModel '
                          'and defined in the lfmc_model module. "LfmcTempCnn" is currently the '
                          'only valid setting.',
        'modelDir':       'A directory for all model outputs',
        'tempDir':        'Directory used to store temporay files such as checkpoints',
        'diagnostics':    'Set to True to display data and model diagnostic details',
        'dataSources':    'A list of the data sources used. If an empty list, the full list of '
                          'data sources the modelClass will process is assumed.',
        'restartRun':     'Used to restart a failed test. Specifies which run to start at.',
        'derivedModels':  'The derived models to create and evaluate. A dictionary where the keys'
                          ' are the model names and the values are a dictionary of parameters, '
                          'including the model type (best, merge, or ensemble). e.g. to create a '
                          'model called "merge10" by merging the last 10 checkpoints, set to '
                          '"{\'merge10\': {\'type\': \'merge\', \'models\': 10}}".  If "falsy", '
                          'no derived models are created. If "truthy" (the default), the standard '
                          'set of derived models are created.',
        'saveModels':     'Set to True, a derived model name or a list of derived model names to '
                          'save the models in h5 format. If True the base model is saved. If '
                          '"falsy", no models are saved.',
        'saveTrain':      'Set to True to save all training output or False to save no training '
                          'output. The default is to save training prediction statistics only.',
        'saveValidation': 'Set to True (default) to save validation predictions. Note: if there '
                          'is validation data, the validation statistics are always saved. '
                          'Ignored if there is no validation data.',
        'plotModel':      'Set to True (default) to create a model plot.',
        'multiSamples':   'If the model is to be trained/evaluated on more than one set of '
                          'training data, a list specifying the key/name for each set of samples. '
                          'If used, each "*Filename" parameter should specify either a single '
                          'file or a list of files the same length as "multiSamples". If the '
                          '"*Filename" parameter is a list of files, the data loaded from each '
                          'file will have a value from "multiSamples" prepended to each row-ID. '
                          'If a single file, the data from the file will be replicated for each '
                          'entry in "multiSamples". Defaults to "None", meaning a single set of '
                          'samples is used, and if any of the "*Filename" parameters are lists, '
                          'the data is concatenated column-wise.',
        'deduplicate':    'Remove duplicates from the training data before training the model. '
                          'For each set of duplicated samples, the target values are adjusted to '
                          'be the mean of the set. Currently this can only be used if a single '
                          'source is specified.',
        'randomSeed':     'Number used to set all random seeds (for random, numpy and tensorflow)',
        'modelRuns':      'Number of times to buid and run the model',
        'resplit':        'True: redo the test/train splits on each run; False: use the same '
                          'test/train split for each run',
        'seedList':       'A list of random seeds used to seed each run if modelRuns > 1. If the '
                          'list size (n) is less than the number of runs, then only the first n '
                          'runs will be seeded. If the list is empty (and modelRuns > 1) the '
                          'randomSeed will be used to seed the first run, all other runs will be '
                          'unseeded. Extra seeds (n > modelRuns) are ignored.',
        'maxWorkers':     'Specifies the maximum number of multiprocess workers to use. Setting '
                          'this > 1 allows parallel processing of folds or runs.',
        'gpuDevice':      'Specifies which GPU device to use. Ignored if "gpuList" is specified.',
        'gpuList':        'Specifies a list of GPU devices. These are assigned to the '
                          'multiprocess workers in a round-robin manner. Each entry in the list '
                          'can be one GPU device number (the worker will use a single GPU) or a '
                          'list of device numbers (the worker will use all these GPUs).',
        'gpuMemory':      'Specifies the GPU memory to use for each worker. The Tensorflow '
                          'default is used if not set or set to a "falsy" value',
        'inputs':         'A dictionary where the keys are the input names and values are '
                          'dictionaries with these keys:\n'
                          'filename:  Required. Full path name of the file containing the data or\n'
                          '           a list of file names. See the "multiSamples" parameter for \n'
                          '           how a list is handled.\n'
                          'channels:  Required for time series inputs. Number of channels in the \n'
                          '           dataset.\n'
                          'normalise: Optional. A dictionary containing the method to use to \n'
                          '           normalise the data, plus any parameters required by this\n'
                          '           method.\n'
                          'start:     Optional. Time series start. The offset from end of the \n'
                          '           input timeseries.\n'
                          'end:       Optional. Time series end. The offset from end of the input\n'
                          '           timeseries. Set to "None" to specify the end of the timeseries.',
        'samplesFile':    'Full path name of the file containing the auxiliary data and target or '
                          'a list of file names. See the "multiSamples" parameter for how a list '
                          'is handled.',
        'auxColumns':     'The columns from the auxilary dataset that should be used as the '
                          'auxiliary input to the model. Either an integer, in which case the'
                          'last auxColumns are used, or a list of the column names to use. The '
                          'columns should not include any columns to be one-hot encoded.',
        'auxAugment':     'Indicates if the auxiliary data should be augmented with the last day '
                          'of the time series data sources. Valid values are "True", "False", or '
                          'a list of data sources to use to augment the auxiliaries.',
        'auxOneHotCols':  'A list of columns in the auxiliary dataset that should be one-hot '
                          'encoded and added to the model auxiliary data. These columns should '
                          'not be included in auxColumns.'
                          'last auxColumns are used, or a list of the column names to use.',
        'targetColumn':   'Column name (in the auxiliary data) of the target column',
        'splitMethod':    '"random" for random train/test splits, "byYear" to split data by '
                          'sample collection year (yearly scenario), "bySite" to split the data '
                          'by sample collection site (out-of-site scenario)',
        'splitSizes':     'A tuple specifying the proportion of data or sites to use for test '
                          'and validation sets for the "random" and "bySite" split methods. If no '
                          'validation set is used, only one value is needed but must be a tuple. '
                          'When using the "byYear" method, this is only relevant when a validation'
                          ' set is wanted; in this case the first value is ignored and the second '
                          'value specifies the proportion of training data to use for the '
                          'validation set.',
        'siteColumn':     'Column name (in the auxiliary data) of the sample collection site.',
        'splitStratify':  'Specifies the column (in the auxiliary data) to use for stratified '
                          'splits, if these are required. Set to False to disable stratified '
                          'splits. Ignored for "byYear" splits.',
        'yearColumn':     'For "byYear" splits, specifies the column (in the auxiliary data) of '
                          'the sample collection year.',
        'splitFolds':     'With "random" or "bySite" splitting: if > 1, k-fold splitting will be '
                          'used. If False, 0 or 1, a single random split will be made. With '
                          '"byYear" splitting: if >= 1, the specified number of yearly splits '
                          'will be used (so if "splitFolds" is 4 and "splitYear" is 2014, 4 '
                          'models will be built, for years 2014, 2015, 2016, and 2017. If False '
                          'or 0, a single split is made and all data for or after the "splitYear" '
                          'is used as test data.',
        'splitYear':      'For "byYear" splits, the first year to year as test data. For "random" '
                          'or "bySite" splits, data in the training set for this year or later is '
                          'discarded, as is data in the test set foe before this year.',
        'trainAdjust':    'For "byYear" splits, The number of days to adjust the end date of the '
                          'training and validation sets. E.g. "90" will remove all samples within '
                          '90 days of the end of the training/validation sets. These samples are '
                          'discarded (NOT added to the test set). The default is 0.',
        'testAdjust':     'For "byYear" splits, the number of days to adjust the start and end '
                          'dates of the test set. A postive number will shift the dates forward '
                          'and a negative number will shift them backwards. E.g. setting'
                          '"test_adjust" to "90", "year" to "2014" and "num_year" to "1" will '
                          'result in a test set containing samples from "01-Apr-2014" to '
                          '"31-Mar-2015". Samples in the adjustment period (e.g. "01-Jan-2014" to '
                          '"31-Mar-2014") are discarded (NOT added to the training or validation '
                          'sets). The default is 0.',
        'convPadding':    'Specifies the Keras padding value for the convolutional layers. Set to '
                          '"valid" for no padding or "same" for padding.',
        'poolPadding':    'Specifies the Keras padding value for the pooling layers. Set to '
                          '"valid" for no padding or "same" for padding.',
        'batchNormalise': "Default setting for including a batch normalisation step in a block, "
                          "can be overriden for a block using the block's bnorm setting",
        'dropoutRate':    "The dropout rate to use for all layers in the model",
        'regulariser':    'A string representation of the regulariser to use in all model layers.'
                          ' If the string starts with "keras", it will be interpreted as a call '
                          'to a keras regularizer function.',
        'validationSet':  'Indicates if a validation set should be used when training the model',
        'earlyStopping':  'If False or 0, early stopping is not used. Otherwise early stopping '
                          'is used and the value used as the patience setting.',
        'epochs':         'Number of training epochs.',
        'batchSize':      'Training batch size.',
        'shuffle':        'Indicates if the training data should be shuffled between epochs.',
        'verbose':        'Sets the verbosity level during training.',
        'optimiser':      'The Keras optimiser. If the value starts with "keras", it will be '
                          'interpreted as code to create a Keras optimizer object.',
        'activation':     'The activation function to use for all model layers',
        'initialiser':    'The function used to initialise the model weights',
        'loss':           'The loss function to use when training the model.',
        'metrics':        'A list of metrics to be evaluated at each checkpoint.',
        'conv':           'A list of convolutional parameter sets, each entry in the list '
                          'corresponds to a convolutional layer that will be added to the model',
        'fc':             'A list of fully connected parameter sets, each entry in the list '
                          'corresponds to a fully connected layer that will be added to the model',
    }


class ExperimentParams(ParamDict):
    """A dictionary for LFMC experiment parameters
    
    A ParamDict sub-class that creates a dictionary with keys for all
    parameters that (together with a set of ModelParams) are needed to
    run an LFMC estimation experiment. Defaults are set for the
    parameters if a source is not provided or has missing keys.
    
    Parameters
    ----------
        source : None, dict, str, or file object, optional
          - If dict: A dictionary containing all model parameters
          - If str: A string representation of the model parameters in
            JSON format.
          - If file object: An open JSON file containing all model
            parameters.
          - If None: The object is initialised with defaults for all
            model parameters.
          - The default is None.
        experiment_name : str, optional
            The name of the experiment. It should be a valid Keras model
            name. The default is 'default_model'.
        
    Attributes
    ----------
    _param_help: dict
        A dictionary containing the ``help`` text for each model
        parameter. The ``general`` key contains the ``help`` text for
        the object.
    """

    def __init__(self, source=None, experiment_name='default_model'):
        super(ExperimentParams, self).__init__(source)
        if source is None:
            self['name'] = experiment_name
        
    def set_defaults(self):
        """Returns the default experiment parameters dictionary
        
        Returns
        -------
        experiment_params : dict
            The dictionary of experiment parameters.
        """
        experiment_params = {
            'name': '',
            'description': '',
            'blocks': {},
            'tests': [{}],
            'testNames': [],
            'restart': False,
            'rerun': [],
            'resumeAllTests': False,
        }
        return experiment_params

    _param_help = {
        'general':        'Dictionary of all parameters used to run an LFMC experiment. A simple '
                          'example of experiment parameters is:\n\n'
                          '{"name": "Filters",\n'
                          ' "description": "Test effect of filter sizes on conv layers",\n'
                          ' "blocks": {\n'
                          '     "conv": {"numLayers": 3, "poolSize": [2, 3, 4]},\n'
                          '     "fc": {}},\n'
                          ' "tests": [\n'
                          '     {"testName": "Filter-8", "conv": {"filters": [8, 8, 8]}},\n'
                          '     {"testName": "Filter-16", "conv": {"filters": [16, 16, 16]}},\n'
                          '     {"testName": "Filter-32", "conv": {"filters": [32, 32, 32]}}],\n'
                          ' "restart": 0}\n\n'
                          'For more help run ExperimentParams().help("parameter").\nAvailable '
                          'parameters are:',
        'name':           'A name for the experiment; must be a valid directory name',
        'description':    'A description of the experiment. Used only for documentation.',
        'blocks':         'A dictionary where the keys are the model blocks used by the '
                          'experiment tests and the values are the block parameters. It must '
                          'include the blocks specified in the tests. Any other blocks are '
                          'optional and any block parameters specified here are ignored. Each '
                          'block parameter should be specified as a list. The first entry in the '
                          'list will be used for the first layer in the block, etc. For each '
                          'block, the layer parameter "numLayers" is required either here or for '
                          'each test and specifies how many layers are required in this block.',
        'tests':          'A list of dictionaries. Each dictionary represents a test and contains '
                          'the changes for the test that need to be made to the experiment model '
                          'parameters. Block parameters should be specified as a list. The first '
                          'entry in the list will be used for the first layer in the block, etc.'
                          'For each block, the layer parameter "numLayers" is required either '
                          'here or on the appropriate "blocks" entry and specifies how many '
                          'layers are required in this block.',
        'testNames':      'A list of the test names. Optional, but if used the list should be '
                          'the same length as "tests".',
        'restart':        'When restarting an experiment, specifies the test number (zero-based) '
                          'at which the experiment run should start. If "None" or "False", the '
                          'experiment will run from the start and a check is made to ensure the '
                          'model directory does not exist.',
        'rerun':          'A list of the test numbers (zero-based) to re-run. If "None", "False" '
                          'or "[]", the experiment will run from the start. Note: if both "rerun" '
                          'and "restart" are specified (i.e. not "None", "False" or "[]"), '
                          '"rerun" will be ignored.',
        'resumeAllTests': 'If "True", any "restartRun" in the model parameters will be applied to '
                          'all tests. If "False" (the default), the "restartRun" will be applied '
                          'to the first test only.',
    }
                