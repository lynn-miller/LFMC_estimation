"""The Model Parameters dictionary"""

import json
import os
import pprint

class ModelParams(dict):
    """A dictionary for LFMC model parameters
    
    Extends the dictionary class by adding a help function. By default,
    the dictionary is created with keys for all parameters needed to
    build a model for LFMC estimation and initialised to the default
    values.
    
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
        conv_layers : int, optional
            The number of convolutional layers to create. See the
            ``set_layers`` method for more details. The default is 0.
        fc_layers : int, optional
            The number of fully connected or dense layers to create.
            See the ``set_layers`` method for more details. The default
            is 0.
        
    Attributes
    ----------
    _param_help: dict
        A dictionary containing the ``help`` text for each model
        parameter. The ``general`` key contains the ``help`` text for
        the object.
    """

    def __init__(self, source=None, model_name='default_model', **blocks):
        if source is None:    # New model - set all parameters to defaults
            model_params = self.set_defaults(model_name)
            super(ModelParams, self).__init__(model_params)
            self.set_layers(**blocks)
        else:
            if type(source) is dict:   # Set parameters from a dictionary
                model_params = source
            elif type(source) is str:  # Set parameters from a JSON string
                model_params = json.loads(source)
            else:                      # Set parameters from a JSON input file
                model_params = json.load(source)
            super(ModelParams, self).__init__(model_params)
        
    def set_defaults(self, model_name='default_model'):
        """Returns the default model parameters dictionary
        
        Parameters
        ----------
        model_name : str, optional
            The name of the model. It should be a valid Keras model
            name. The default is 'default_model'.

        Returns
        -------
        model_params : dict
            The dictionary of model parameters.
        """
        model_params = {
            'modelName': model_name,
            'description': '',
            'modelClass': 'LfmcModel',
            'modelDir': '',
            'tempDir': '',
            'diagnostics': False,
            'dataSources': [],
            'restartRun': None,
            'saveModels': False,
            'saveTrain': None,
            'plotModel': True,

            'randomSeed': 1234,
            'modelSeed': 1234,
            'modelRuns': 1,
            'resplit': False,
            'seedList': [],
            
            # Multiprocessing parameters
            'maxWorkers': 1,
            'deterministic': False,
            'gpuDevice': 0,
            'gpuMemory': 0,

            # MODIS data parameters
            'modisFilename': None,
            'modisChannels': 7,
            'modisNormalise': {'method': 'minMax', 'percentiles': 2},

            # PRISM data parameters
            'prismFilename': None,
            'prismChannels': 7,
            'prismNormalise': {'method': 'minMax', 'percentiles': 2},

            # Auxiliary data parameters
            'auxFilename': None,
            'auxColumns': 9,
            'auxAugment': True,
            'auxOneHotCols': ['Koppen'],
            'targetColumn': 'LFMC value',

            # Data splitting parameters
            'splitMethod': 'byYear',
            'splitSizes': (0.33, 0.067),
            'siteColumn': 'Site',
            'splitStratify': 'Land Cover',
            'splitYear': None,
            'yearColumn': 'Sampling year',
            'splitFolds': 0,
            
            # Keras common parameters
            'convPadding': 'same',
            'poolPadding': 'valid',

            # Overfitting controls
            'batchNormalise': True,
            'dropoutRate': 0.3,
            'regulariser': 'keras.regularizers.l2(1.e-6)',
            'validationSet': False,
            'earlyStopping': False,

            # Fitting parameters
            'epochs': 100,
            'evaluateEpochs': None,
            'batchSize': 64,
            'shuffle': True,
            'verbose': 0,

            # Keras methods
            'optimiser': 'adam',
            'activation': 'relu',
            'initialiser': 'he_normal',
            'loss': 'mean_squared_error',
            'metrics': ['mean_absolute_error'],
        }
        return model_params

    def __str__(self):
        return pprint.pformat(self, width=100, sort_dicts=False)

    def set_layers(self, **kwargs):
        """Sets model layer parameters
        
        Sets the parameters for the convolutional (``conv``) and fully
        connected (``fc``) layers.

        Parameters
        ----------
        conv_layers : int, optional
            The number of convolutional layers. The default is None.
        fc_layers : int, optional
            The number of fully connected layers. The default is None.

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

    def save(self, file_stream=None):
        """Saves the model parameters
        
        Convert the model parameters dictionary to a JSON string and
        optionally save to a file.
        
        Parameters
        ----------
        file_stream : str or file handle, optional
            Either the name of the output file, or a file handle for
            the output file. If None, the converted JSON string is
            returned. The default is None.

        Returns
        -------
        str
            The JSON representation of the model parameters. Only
            returned if no file stream parameter specified.

        """
        if file_stream is None:               # No output file, return parameters as a JSON string
            return json.dumps(self, indent=2)
        elif isinstance(file_stream, str):    # File name provided, open file and save parameters
            if not os.path.exists(self['modelDir']):
                os.makedirs(self['modelDir'])
            with open(os.path.join(self['modelDir'], file_stream), 'w') as f:
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
                        "'", "").replace("\n ", f"\n{spaces}").replace("\\n", "\n")
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
        'general':        'Dictionary of all parameters used to build an LFMC model. For more '
                          'help run model_params.help("parameter").\nAvailable parameters are:',
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
        'saveModels':     'Set to True to save the models in h5 format.',
        'saveTrain':      'Set to True to save all training output or False to save no training '
                          'output. The default is to save training prediction statistics only.',
        'plotModel':      'Set to True (default) to create a model plot.',
        'randomSeed':     'Number used to set all random seeds (for random, numpy and tensorflow)',
        'modelRuns':      'Number of times to buid and run the model',
        'resplit':        'True: redo the test/train splits on each run; False: use the same '
                          'test/train split for each run',
        'seedList':       'A list of random seeds used to seed each run if modelRuns > 1. If the '
                          'list size (n) is less than the number of runs, then only the first n '
                          'runs will be seeded. If the list is empty (and modelRuns > 1) the '
                          'randomSeed will be used to seed the first run, all other runs will be '
                          'unseeded. Extra seeds (n > modelRuns) are ignored.',
        'maxWorkers':     'Specifies the maximum number of workers to use. Setting this > 1 allows'
                          ' parallel processing of folds or runs.',
        'gpuDevice':      'Specifies which GPU device to use. Ignored if "gpuMemory" is not set or'
                          ' set to a "falsy" value.',
        'gpuMemory':      'Specifies the GPU memory to use for each sub-process. The Tensorflow '
                          'default is used if not set or set to a "falsy" value',
        'modisFilename':  'Full path name of the file containing the MODIS data.',
        'modisChannels':  'Number of channels in the MODIS dataset.',
        'modisNormalise': 'A dictionary containing the method to use to normalise the MODIS '
                          'data, plus any parameters required by this method.',
        'modisStart':     'MODIS timeseries start. The offset from end of the input timeseries.',
        'modisEnd':       'MODIS timeseries end. The offset from end of the input timeseries. '
                          'Set to "None" to specify the end of the timeseries.',
        'prismFilename':  'Full path name of the file containing the PRISM data',
        'prismChannels':  'Number of channels in the PRISM dataset',
        'prismNormalise': 'A dictionary containing the method to use to normalise the PRISM '
                          'data, plus any parameters required by this method',
        'prismStart':     'PRISM timeseries start. The offset from end of the input timeseries. ',
        'prismEnd':       'PRISM timeseries end. The offset from end of the input timeseries. '
                          'Set to "None" to specify the end of the timeseries.',
        'auxFilename':    'Full path name of the file containing the auxiliary data and target',
        'auxColumns':     'The columns from the auxilary dataset that should be used as the '
                          'auxiliary input to the model. Either an integer, in which case the'
                          'last auxColumns are used, or a list of the column names to use. The '
                          'columns should not include any columns to be one-hot encoded.',
        'auxAugment':     'Indicates if the auxiliary data should be augmented with the last day '
                          'of all time series data sources.',
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
        