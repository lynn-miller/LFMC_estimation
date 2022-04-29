""" Modis-tempCNN Architecture
Defines the model parameters for the Modis-tempCNN architecture.
""" 

import initialise
from model_parameters import ModelParams


# ===================================================
# Model parameters for the Modis-tempCNN architecture
# ===================================================
model_params = ModelParams(modis_layers=3, fc_layers=2)

model_params['modelClass'] = 'LfmcTempCnn'
model_params['dataSources'] = ['modis', 'aux']
model_params['auxColumns'] = ['Day_sin', 'Day_cos', 'Long_sin', 'Long_cos', 'Lat_norm',
                              'Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos']
model_params['auxOneHotCols'] = []
model_params['auxAugment'] = False
model_params['dropoutRate'] = 0.5
model_params['batchSize'] = 32
model_params['epochs'] = 100
# Modis-tempCNN isn't an ensemble so only 1 run is needed
model_params['modelRuns'] = 1
# Modis-tempCNN uses the model formed by merging the last 10 checkpoints
model_params['derivedModels'] = {'merge10': {'type': 'merge', 'models': 10}}
    
model_params['fc'][0]['units'] = 256
model_params['fc'][1]['units'] = 256

model_params['modisConv'][0]['filters'] = 32
model_params['modisConv'][1]['filters'] = 32
model_params['modisConv'][2]['filters'] = 32

model_params['modisConv'][0]['poolSize'] = 2
model_params['modisConv'][1]['poolSize'] = 3
model_params['modisConv'][2]['poolSize'] = 4
