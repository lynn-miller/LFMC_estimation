""" LFMC Within-site Architecture
Defines the model parameters for the within-site architecture.
""" 

import initialise
import common
from model_parameters import ModelParams


# =================================================
# Model parameters for the within-site architecture
# =================================================
model_params = ModelParams(modis_layers=5, prism_layers=5, fc_layers=3)

model_params['modelClass'] = 'LfmcTempCnn'
model_params['dataSources'] = ['modis', 'prism', 'aux'] 
model_params['auxColumns'] = ['Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos',
                              'Long_sin', 'Long_cos', 'Lat_norm']
model_params['auxOneHotCols'] = ['Czone3']
model_params['auxAugment'] = True
model_params['dropoutRate'] = 0.1
model_params['epochs'] = 100
model_params['batchSize'] = 512
model_params['modelRuns'] = common.ENSEMBLE_SIZE

model_params['fc'][0]['units'] = 512
model_params['fc'][1]['units'] = 512
model_params['fc'][2]['units'] = 512

model_params['modisConv'][0]['poolSize'] = 0
model_params['modisConv'][1]['poolSize'] = 5
model_params['modisConv'][2]['poolSize'] = 2
model_params['modisConv'][3]['poolSize'] = 3
model_params['modisConv'][4]['poolSize'] = 4

model_params['modisConv'][0]['filters'] = 8
model_params['modisConv'][1]['filters'] = 8
model_params['modisConv'][2]['filters'] = 8
model_params['modisConv'][3]['filters'] = 8
model_params['modisConv'][4]['filters'] = 8

model_params['prismConv'][0]['poolSize'] = 0
model_params['prismConv'][1]['poolSize'] = 5
model_params['prismConv'][2]['poolSize'] = 2
model_params['prismConv'][3]['poolSize'] = 3
model_params['prismConv'][4]['poolSize'] = 4

model_params['prismConv'][0]['filters'] = 8
model_params['prismConv'][1]['filters'] = 8
model_params['prismConv'][2]['filters'] = 8
model_params['prismConv'][3]['filters'] = 8
model_params['prismConv'][4]['filters'] = 8
