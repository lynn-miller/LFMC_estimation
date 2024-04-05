""" Modis-tempCNN Architecture
Defines the model parameters for the Modis-tempCNN architecture.
""" 

import initialise
from model_parameters import ModelParams


# ===================================================
# Model parameters for the Modis-tempCNN architecture
# ===================================================
model_params = ModelParams(optical_layers=3, fc_layers=2)

model_params['modelClass'] = 'LfmcTempCnn'
# Modis-tempCNN isn't an ensemble so only 1 run is needed
model_params['modelRuns'] = 1
model_params['dataSources'] = ['optical', 'aux']
# Modis-tempCNN uses the model formed by merging the last 10 checkpoints
model_params['derivedModels'] = {'merge10': {'type': 'merge', 'models': 10}}

# Target column
model_params['targetColumn'] = 'LFMC value'

# Auxiliary data
model_params['auxColumns'] = ['Day_sin', 'Day_cos', 'Long_sin', 'Long_cos', 'Lat_norm',
                              'Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos']
model_params['auxOneHotCols'] = []
model_params['auxAugment'] = False

# Keras common parameters
model_params['convPadding'] = 'same'
model_params['poolPadding'] = 'valid'

# Overfitting controls
model_params['batchNormalise'] = True
model_params['dropoutRate'] = 0.5
model_params['regulariser'] = 'keras.regularizers.l2(1.e-6)'

# Fitting parameters
model_params['epochs'] = 100
model_params['batchSize'] = 32
model_params['shuffle'] = True

# Keras methods
model_params['optimiser'] = 'adam'
model_params['activation'] = 'relu'
model_params['initialiser'] = 'he_normal'
model_params['loss'] = 'mean_squared_error'
model_params['metrics'] = ['mean_absolute_error']

# Block / layer parameters
model_params['blocks']['fc'][0].update({'units': 256, 'bnorm': True})
model_params['blocks']['fc'][1].update({'units': 256, 'bnorm': True})

model_params['blocks']['opticalConv'][0].update(
    {'poolSize': 2, 'filters': 32, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][1].update(
    {'poolSize': 3, 'filters': 32, 'kernel': 5, 'bnorm': True})
model_params['blocks']['opticalConv'][2].update(
    {'poolSize': 4, 'filters': 32, 'kernel': 5, 'bnorm': True})