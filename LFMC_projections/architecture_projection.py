""" LFMC Out-of-site Architecture
Defines the model parameters for the out-of-site architecture.
""" 

import initialise
import common
from model_parameters import ModelParams


# ================================================
# Model parameters for the projection architecture
# ================================================
model_params = ModelParams(modis_layers=3, prism_layers=3, fc_layers=1)

model_params['modelClass'] = 'LfmcTempCnn'
model_params['modelRuns'] = common.ENSEMBLE_SIZE
model_params['dataSources'] = ['modis', 'prism', 'aux'] 

# Target column
model_params['targetColumn'] = 'LFMC value'

# Auxiliary data
model_params['auxColumns'] = ['Long_sin', 'Long_cos', 'Lat_norm']
model_params['auxOneHotCols'] = ['Czone3']
model_params['auxAugment'] = False

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
model_params['fc'][0]['units'] = 128
model_params['fc'][0]['bnorm'] = True

model_params['modisConv'][0]['poolSize'] = 2
model_params['modisConv'][1]['poolSize'] = 3
model_params['modisConv'][2]['poolSize'] = 4

model_params['modisConv'][0]['filters'] = 8
model_params['modisConv'][1]['filters'] = 8
model_params['modisConv'][2]['filters'] = 8

model_params['modisConv'][0]['kernel'] = 5
model_params['modisConv'][1]['kernel'] = 5
model_params['modisConv'][2]['kernel'] = 5

model_params['modisConv'][0]['bnorm'] = True
model_params['modisConv'][1]['bnorm'] = True
model_params['modisConv'][2]['bnorm'] = True

model_params['prismConv'][0]['poolSize'] = 2
model_params['prismConv'][1]['poolSize'] = 3
model_params['prismConv'][2]['poolSize'] = 4

model_params['prismConv'][0]['filters'] = 8
model_params['prismConv'][1]['filters'] = 8
model_params['prismConv'][2]['filters'] = 8

model_params['prismConv'][0]['kernel'] = 5
model_params['prismConv'][1]['kernel'] = 5
model_params['prismConv'][2]['kernel'] = 5

model_params['prismConv'][0]['bnorm'] = True
model_params['prismConv'][1]['bnorm'] = True
model_params['prismConv'][2]['bnorm'] = True
