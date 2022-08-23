# Projecting live fuel moisture content via deep learning
The iPython notebooks and Python scripts in the sub-directories can be used to extract data from Google Earth Engine and run the scenarios from the paper.

## Data
Seven data sources are used:
1. <a name="Globe-LFMC">The `Globe-LFMC.xlsx` datatset - https://springernature.figshare.com/collections/Globe-LFMC_a_global_plant_water_status_database_for_vegetation_ecophysiology_and_wildfire_applications/4526810/2</a>
2. MODIS MCD43A4.006 Nadir BRDF-Adjusted Reflectance Daily 500m data. Google Earth Engine product: MODIS/006/MCD43A4 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4
3. PRISM AN81d Gridded daily climate dataset. Google Earth Engine product: OREGONSTATE/PRISM/AN81d - https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81d
4. <a name="Koppen-Geiger">Present-day (1980–2016) Köppen-Geiger climate classification dataset - https://figshare.com/articles/dataset/Present_and_future_K_ppen-Geiger_climate_classification_maps_at_1-km_resolution/6396959/2</a>
5. The NASA SRTM Digital Elevation 30m data. Google Earth Engine product: USGS/SRTMGL1_003 - https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
6. MODIS MOD44W.006 MODIS water mask. Google Earth Engine product: MODIS/006/MOD44W - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD44W

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
### Modis-tempCNN architecture 
The architecture file defines the Multi-tempCNN architectures used for projections:
- Multi-tempCNN projection architecture: `architecture_projection.py`

### Experiment Models
The `evaluation_models.py` script runs an experiment that builds a pool of 50 model sets for each of the lead times (nowcasting to 12 months lead time at one-month increments).

The `evaluation_ensembles.ipynb` notebooks uses the model set pools to generate the sets of ensembles used to evaluate the projections models.

### LFMC map models
The models used to generate the <a name="maps">LFMC maps</a> are created by `final_models.ipynb`. This creates both the nowcasting and projection models used to create the LFMC maps.

## Analyse Results
### Generate Results and Figures
The notebooks to calculate the evalation metrics and generate the figures are:
| Notebook | Relevant section in paper | Description |
| ---- | ---- | ---- |
| `Lead time analysis.ipynb` | Analysis of model results over increasing lead times |  |
| `Time series and lead times.ipynb` | Analysis of model results over increasing lead times |  |
| `Land cover analysis.ipynb` | Model performance by land cover and elevation; Ability of the model to identify high fire risk conditions |  |
| `LFMC range analysis.ipynb` | Model performance by LFMC range |  |
| `Climate zone analysis.ipynb` | Model performance by climate zone |  |
| `Location analysis.ipynb` | Model performance by sampling site locations |  |

## Create Maps
Run the notebooks in the `create_maps` in the following order to create the LFMC and uncertainty maps. This assumes the [mapping models](#maps) have been created.
### Extract modelling data from Google Earth Engine
1. `Extract Auxiliary Data Conus.ipynb`
2. `Extract MODIS Conus Image.ipynb`
3. `Extract PRISM Conus Image.ipynb`
### Create the TIFFs used to generate maps
4. `CONUS LFMC maps.ipynb`

Maps can be created from the output TIFFs using a GIS tool such as QGIS. 