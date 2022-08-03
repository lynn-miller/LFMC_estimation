""" LFMC Out-of-site Architecture
Defines the model parameters for the out-of-site architecture.
""" 

import initialise
import common
from model_parameters import ModelParams


# =================================================
# Model parameters for the out-of-site architecture
# =================================================
model_params = ModelParams(modis_layers=3, prism_layers=3, fc_layers=1)

model_params['modelClass'] = 'LfmcTempCnn'
model_params['dataSources'] = ['modis', 'prism', 'aux'] 
model_params['auxColumns'] = ['Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos',
                              'Long_sin', 'Long_cos', 'Lat_norm']
model_params['auxOneHotCols'] = ['Czone3']
model_params['auxAugment'] = True
model_params['dropoutRate'] = 0
model_params['epochs'] = 50
model_params['batchSize'] = 512
model_params['modelRuns'] = common.ENSEMBLE_SIZE

model_params['fc'][0]['units'] = 128

model_params['modisConv'][0]['poolSize'] = 2
model_params['modisConv'][1]['poolSize'] = 3
model_params['modisConv'][2]['poolSize'] = 4

model_params['modisConv'][0]['filters'] = 8
model_params['modisConv'][1]['filters'] = 8
model_params['modisConv'][2]['filters'] = 8

model_params['prismConv'][0]['poolSize'] = 2
model_params['prismConv'][1]['poolSize'] = 3
model_params['prismConv'][2]['poolSize'] = 4

model_params['prismConv'][0]['filters'] = 8
model_params['prismConv'][1]['filters'] = 8
model_params['prismConv'][2]['filters'] = 8
