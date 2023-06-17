# Inter-Continental Domain Adaptation for Temporal CNNs
The iPython notebooks and Python scripts in the sub-directories can be used to extract data from Google Earth Engine and run the scenarios from the paper.

## Data
Four data sources are used:
1. <a name="Globe-LFMC">The `Globe-LFMC.xlsx` datatset - https://springernature.figshare.com/collections/Globe-LFMC_a_global_plant_water_status_database_for_vegetation_ecophysiology_and_wildfire_applications/4526810/2</a>
2. MODIS MCD43A4.006 Nadir BRDF-Adjusted Reflectance Daily 500m data. Google Earth Engine product: MODIS/006/MCD43A4 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4
3. ERA5 Gridded daily climate dataset. Google Earth Engine product: ECMWF/ERA5/DAILY - https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY
4. <a name="Koppen-Geiger">Present-day (1980–2016) Köppen-Geiger climate classification dataset - https://figshare.com/articles/dataset/Present_and_future_K_ppen-Geiger_climate_classification_maps_at_1-km_resolution/6396959/2</a>

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
Run the notebooks in the `extract_data` directory in the following order:
### Create the CONUS datasets
1. `CONUS Auxiliary Data.ipynb`
2. `CONUS MODIS Data.ipynb`
3. `CONUS ERA5 Data.ipynb`
4. `CONUS Locations.ipynb`
### Create the Australian datasets
1. `Australia Auxiliary Data.ipynb`
2. `Australia MODIS Data.ipynb`
3. `Australia ERA5 Data.ipynb`
4. `Australia Locations.ipynb`
### Create the European datasets
1. `Europe Auxiliary Data.ipynb`
2. `Europe MODIS Data.ipynb`
3. `Europe ERA5 Data.ipynb`
4. `Europe Locations.ipynb`

## Build Models
Run the notebooks and scripts in the `build_models` directory in the following order:

### CONUS Source Models
Create the CONUS source models by running `conus_base_models.py`.

### Generate fold for Australian and European tests
1. Create the Australian folds by running `australia_gen-folds.py`.
2. Create the European folds by running `europe_gen-folds.py`.

### Experiment Models
There is a script or notebook that can be run to build or adapt models as required for each test, named `<region_size>_<method>.{py}{ipynb}`

## Analyse Results

### Summarize Results
The notebooks `<region_size>_results_summ.ipynb` create summary csv files of the predictions and statistics for each test.

The `results` directory contain copies of these files generated for the tests used to obtain the results in the paper.

### Analyse results and create figures
The notebooks `Australia_analysis.ipynb` and `Europe_analysis.ipynb` display a summary of the results statistics and create figures 8 and 9 from the paper.
