""" LFMC Out-of-site Architecture
Defines the model parameters for the out-of-site architecture.
""" 

import initialise
import common
from model_parameters import ModelParams


# =======================================================
# Model parameters for the transfer learning architecture
# =======================================================
model_params = ModelParams(opticalConv_layers=3, weatherConv_layers=3, fc_layers=1)

model_params['modelClass'] = 'LfmcTempCnn'
model_params['modelRuns'] = common.ENSEMBLE_SIZE
model_params['dataSources'] = ['optical', 'weather', 'aux'] 

# Target column
model_params['targetColumn'] = 'LFMC value'
model_params['targetNormalise'] = {'method': 'standard'}

# Auxiliary data
model_params['auxColumns'] = ['Long_sin', 'Long_cos', 'Lat_norm']
model_params['auxOneHotCols'] = ['Czone3']
model_params['auxAugment'] = True

# Keras common parameters
model_params['convPadding'] = 'same'
model_params['poolPadding'] = 'valid'

# Overfitting controls
model_params['batchNormalise'] = True
model_params['dropoutRate'] = 0
model_params['regulariser'] = 'keras.regularizers.l2(1.e-6)'

# Fitting parameters
model_params['epochs'] = 50
model_params['batchSize'] = 512
model_params['shuffle'] = True

# Keras methods
model_params['optimiser'] = 'adam'
model_params['activation'] = 'relu'
model_params['initialiser'] = 'he_normal'
model_params['loss'] = 'mean_squared_error'
model_params['metrics'] = ['mean_absolute_error']

# Block / layer parameters
model_params['blocks']['fc'][0].update({'units': 128, 'bnorm': True})

model_params['blocks']['opticalConv'][0].update(
    {'poolSize': 2, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][1].update(
    {'poolSize': 3, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][2].update(
    {'poolSize': 4, 'filters': 8, 'kernel': 5, 'bnorm': True})

model_params['blocks']['weatherConv'][0].update(
    {'poolSize': 2, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['weatherConv'][1].update(
    {'poolSize': 3, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['weatherConv'][2].update(
    {'poolSize': 4, 'filters': 8, 'kernel': 5, 'bnorm': True})
