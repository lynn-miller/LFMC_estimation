""" LFMC Within-site Architecture
Defines the model parameters for the within-site architecture.
""" 

import initialise
import common
from model_parameters import ModelParams


# =================================================
# Model parameters for the within-site architecture
# =================================================
model_params = ModelParams(optical_layers=5, weather_layers=5, fc_layers=3)

model_params['modelClass'] = 'LfmcTempCnn'
model_params['modelRuns'] = common.ENSEMBLE_SIZE
model_params['dataSources'] = ['optical', 'weather', 'aux'] 

# Target column
model_params['targetColumn'] = 'LFMC value'

# Auxiliary data
model_params['auxColumns'] = ['Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos',
                              'Long_sin', 'Long_cos', 'Lat_norm']
model_params['auxOneHotCols'] = ['Czone3']
model_params['auxAugment'] = True

# Keras common parameters
model_params['convPadding'] = 'same'
model_params['poolPadding'] = 'valid'

# Overfitting controls
model_params['batchNormalise'] = True
model_params['dropoutRate'] = 0.1
model_params['regulariser'] = 'keras.regularizers.l2(1.e-6)'

# Fitting parameters
model_params['epochs'] = 100
model_params['batchSize'] = 512
model_params['shuffle'] = True

# Keras methods
model_params['optimiser'] = 'adam'
model_params['activation'] = 'relu'
model_params['initialiser'] = 'he_normal'
model_params['loss'] = 'mean_squared_error'
model_params['metrics'] = ['mean_absolute_error']

# Block / layer parameters
model_params['blocks']['fc'][0].update({'units': 512, 'bnorm': True})
model_params['blocks']['fc'][1].update({'units': 512, 'bnorm': True})
model_params['blocks']['fc'][2].update({'units': 512, 'bnorm': True})

model_params['blocks']['opticalConv'][0].update(
    {'poolSize': 0, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][1].update(
    {'poolSize': 5, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][2].update(
    {'poolSize': 2, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][3].update(
    {'poolSize': 3, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][4].update(
    {'poolSize': 4, 'filters': 8, 'kernel': 5, 'bnorm': True})

model_params['blocks']['weatherConv'][0].update(
    {'poolSize': 0, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['weatherConv'][1].update(
    {'poolSize': 5, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['weatherConv'][2].update(
    {'poolSize': 2, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['weatherConv'][3].update(
    {'poolSize': 3, 'filters': 8, 'kernel': 5, 'bnorm': True})
model_params['blocks']['weatherConv'][4].update(
    {'poolSize': 4, 'filters': 8, 'kernel': 5, 'bnorm': True})
