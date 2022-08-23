# Multi-modal Temporal CNNs for Live Fuel Moisture Content Estimation
The iPython notebooks and Python scripts in the sub-directories can be used to extract data from Google Earth Engine and run the scenarios from the paper.

## Data
Seven data sources are used:
1. <a name="Globe-LFMC">The `Globe-LFMC.xlsx` datatset - https://springernature.figshare.com/collections/Globe-LFMC_a_global_plant_water_status_database_for_vegetation_ecophysiology_and_wildfire_applications/4526810/2</a>
2. The NASA SRTM Digital Elevation 30m data. Google Earth Engine product: USGS/SRTMGL1_003 - https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
3. MODIS MCD43A4.006 Nadir BRDF-Adjusted Reflectance Daily 500m data. Google Earth Engine product: MODIS/006/MCD43A4 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4
4. MODIS MOD10A1.006 Terra Snow Cover Daily Global 500m data. Google Earth Engine product: MODIS/006/MOD10A1 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD10A1
5. MODIS MOD44W.006 MODIS water mask. Google Earth Engine product: MODIS/006/MOD44W - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD44W
6. PRISM AN81d Gridded daily climate dataset. Google Earth Engine product: OREGONSTATE/PRISM/AN81d - https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81d
7. <a name="Koppen-Geiger">Present-day (1980–2016) Köppen-Geiger climate classification dataset - https://figshare.com/articles/dataset/Present_and_future_K_ppen-Geiger_climate_classification_maps_at_1-km_resolution/6396959/2</a>

## Set Up
The `common.py` file sets up global variables and parameters used by the notebooks and scripts.

### Directory and File Settings to check/change
The first two parameters in `common.py` should be updated as necessary
```
# Top-level directory for all LFMC data
LFMC_DATA_DIR = r'G:\My Drive\LFMC Data'   <--- All data and model outputs are stored in sub-directories of this location

# Temporary directory
TEMP_DIR = r'C:\Temp\LFMC'   <--- Used for storing temporary files such as Tensorflow checkpoint files
```
None of the other directory and file names *need* to be changed, but can be if desired.

### Other Parameters and Settings
Leave all other parameters and setting in `common.py` as they are to replicate results in the paper.

## Extract Data
Run the three notebooks in the `extract_data` directory in the following order:
1. `Extract Auxiliary Data.ipynb`
2. `Extract MODIS Data.ipynb`
3. `Extract PRISM Data.ipynb`

## Build Models
### Multi-tempCNN and Modis-tempCNN architecture 
The three architecture files define the Multi-tempCNN architectures presented in the paper, plus the Modis-tempCNN baseline architecture:
- Multi-tempCNN out-of-site architecture: `architecture_out_of_site.py`
- Multi-tempCNN within-site architecture: `architecture_within_site.py`
- Modis-tempCNN architecture: `architecture_modis_tempCNN.py`

### Experiment Models
There is a notebook or script in the `build_models` directory to build the models used in each of the experiments in the paper. Each of these scripts builds a pool of 50 model sets. These model set pools are used by the [ensemble creation](#ensemble-creation) scripts to generate the sets of ensembles used to evaluate the Multi-tempCNN models.

Most of the results are from the <a name="main">main Multi-tempCNN models</a> created by:
- Out-of-site model set pool: `out-of-site_models.py`
- Within-site model set pool: `within-site_models.py`

To create the model set pools for the comparison tests (section 3.1.4 in the paper) run <a name="comp">`comparison models.ipynb`</a>.

To create the model set pools for the <a name="input">input ablation tests</a> (section 3.3.1 in the paper) run:
- `out-of-site_omit_one.py`
- `within-site_omit_one.py`

To create the model set pools for the architecture <a name="ablation">ablation tests</a> (section 3.3.2 in the paper) run:
- `out-of-site_ablation.py`
- `within-site_ablation.py`

Notebooks used to determine the run times reported in sections 3.1.5 and 3.3.2 of the paper:
- `out-of-site run times.ipynb`
- `within-site run times.ipynb`

### LFMC map models
The models used to generate the <a name="maps">LFMC maps</a> are created by:
- Modis-tempCNN model: `map2017_modis_tempCNN.ipynb`
- Multi-tempCNN out-of-site model: `map2017_out-of-site_model.ipynb`

## Analyse Results
### Ensemble Creation
There are a set of notebooks to generate the ensembles from the model set pools:
- <a name="ens-main">`Generate main ensembles.ipynb` creates ensembles from the [main out-of-site and within-site model set pools](#main).</a>
- <a name="ens-comp">`Generate comparison ensembles.ipynb` creates ensembles from each of the [comparison model set pools](#comp).</a>
- <a name="ens-change">`Generate changes ensembles.ipynb` creates ensembles from the [input ablation](#input) and [architecture ablation](#ablation)</a> model set pools.
### Generate Results and Figures
The notebooks to calculate the evalation metrics and generate the figures are:
| Notebook | Relevant section in paper | Pre-requisites |
| ---- | ---- | ---- |
| `Main result.ipynb` | 3.1 | [Main ensembles](#ens-main) |
| `Comparison models.ipynb` | 3.1.4 | [Main ensembles](#ens-main), [Comparison ensembles](#ens-comp) |
| `Ensembling results.ipynb` | 3.2 | [Main ensembles](#ens-main) |
| `Ablation tests.ipynb` | 3.3 | [Main ensembles](#ens-main), [Changes ensembles](#ens-change) |
| `Landcover evaluation.ipynb` | 3.4.1, 3.4.2| [Main ensembles](#ens-main) |
| `LFMC range evaluation.ipynb` | 3.4.3 | [Main ensembles](#ens-main) |
| `Climate zone analysis.ipynb` | 3.5.1 | [Main ensembles](#ens-main) |
| `Locations analysis.ipynb` | 3.5.2 | [Main ensembles](#ens-main) |
| `Compare maps.ipynb` | 3.5.3 | [Mapping models](#maps), <br>plus a TIFF that is the difference between the out-of-site TIFF <br>and the Modis-tempCNN TIFF (code not provided) |

## Create Maps
Run the notebooks in the `create_maps` in the following order to create the LFMC and uncertainty maps. This assumes the [mapping models](#maps) have been created.
### Extract modelling data from Google Earth Engine
1. `Extract Auxiliary Data Conus.ipynb`
2. `Extract MODIS Conus Image.ipynb`
3. `Extract PRISM Conus Image.ipynb`
### Create the TIFFs used to generate maps
4. `CONUS LFMC maps.ipynb`

Maps can be created from the output TIFFs using a GIS tool such as QGIS. 