# Multi-modal Temporal CNNs for Live Fuel Moisture Content Estimation
This repository contains code to build and evaluate temporal CNNs for Live Fuel Moisture Content Estimation (LFMC). These models (Multi-tempCNN) use time series of reflectance and meteorological data, together with some auxiliary variables to train the CNNs to estimate LFMC. The code reproduces the results in the paper "Multi-modal Temporal CNNs for Live Fuel Moisture Content Estimation" (submitted for publication).

Note that due to random variable seed settings and randomness in the scheduling of Tensorflow tasks on GPUs, results obtained may differ slightly from those shown in the paper, but should be broadly similar.

## Data
Seven data sources are used:
<a name="Globe-LFMC">
1. The `Globe-LFMC.xlsx` datatset - https://springernature.figshare.com/collections/Globe-LFMC_a_global_plant_water_status_database_for_vegetation_ecophysiology_and_wildfire_applications/4526810/2</a>
2. The NASA SRTM Digital Elevation 30m data. Google Earth Engine product: USGS/SRTMGL1_003 - https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
3. MODIS MCD43A4.006 Nadir BRDF-Adjusted Reflectance Daily 500m data. Google Earth Engine product: MODIS/006/MCD43A4 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4
4. MODIS MOD10A1.006 Terra Snow Cover Daily Global 500m data. Google Earth Engine product: MODIS/006/MOD10A1 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD10A1
5. MODIS MOD44W.006 MODIS water mask. Google Earth Engine product: MODIS/006/MOD44W - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD44W
6. PRISM AN81d Gridded daily climate dataset. Google Earth Engine product: OREGONSTATE/PRISM/AN81d - https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81d
7. <a name="Koppen-Geiger">Present-day (1980–2016) Köppen-Geiger climate classification dataset - https://figshare.com/articles/dataset/Present_and_future_K_ppen-Geiger_climate_classification_maps_at_1-km_resolution/6396959/2</a>

## Requirements
1. Python 3.8 with packages listed in [requirements.txt](../requirements.txt). The code was written and tested using Anaconda on Windows. Limited testing has also been done on Linux, but there were minor differences in the versions of some packages (see [requirements-linux.txt](../requirements-linux.txt) for package versions used). [InstallNotes.md](../InstallNotes.md) contains some notes about how to set up a suitable Anaconda virtual environment.
2. An authenticated Google Earth Engine account
3. A copy of the [`Globe-LFMC.xlsx` dataset](#Globe-LFMC)
4. A copy of the [Köppen-Geiger Climate Zone Data](#Koppen-Geiger)

## Code
The iPython notebooks and Python scripts in can be used to extract data from Google Earth Engine and run the scenarios from the paper.

### Set Up
The `common.py` file sets up global variables and parameters used by the notebooks and scripts.

#### Directory and File Settings to check/change
The first two parameters in `common.py` should be updated as necessary
```
# Top-level directory for all LFMC data
LFMC_DATA_DIR = r'G:\My Drive\LFMC Data'   <--- All data and model outputs are stored in sub-directories of this location

# Temporary directory
TEMP_DIR = r'C:\Temp\LFMC'   <--- Used for storing temporary files such as Tensorflow checkpoint files
```
None of the other directory and file names *need* to be changed, but can be if desired.

#### Other Parameters and Settings
Leave all other parameters and setting in `common.py` as they are to replicate results in the paper.

### Extract Data
Run the three notebooks in the `extract_data` directory in the following order:
1. `Extract Auxiliary Data.ipynb`
2. `Extract MODIS Data.ipynb`
3. `Extract PRISM Data.ipynb`

### Build Models
#### Multi-tempCNN and Modis-tempCNN architecture 
The three architecture files define the Multi-tempCNN architectures presented in the paper, plus the Modis-tempCNN baseline architecture:
- Multi-tempCNN out-of-site architecture: `architecture_out_of_site.py`
- Multi-tempCNN within-site architecture: `architecture_within_site.py`
- Modis-tempCNN architecture: `architecture_modis_tempCNN.py`

#### Experiment Models
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

#### LFMC map models
The models used to generate the <a name="maps">LFMC maps</a> are created by:
- Modis-tempCNN model: `map2017_modis_tempCNN.ipynb`
- Multi-tempCNN out-of-site model: `map2017_out-of-site_model.ipynb`

### Analyse Results
#### Ensemble Creation
There are a set of notebooks to generate the ensembles from the model set pools:
- <a name="ens-main">`Generate main ensembles.ipynb` creates ensembles from the [main out-of-site and within-site model set pools](#main).</a>
- <a name="ens-comp">`Generate comparison ensembles.ipynb` creates ensembles from each of the [comparison model set pools](#comp).</a>
- <a name="ens-change">`Generate changes ensembles.ipynb` creates ensembles from the [input ablation](#input) and [architecture ablation](#ablation)</a> model set pools.
#### Generate Results and Figures
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

### Create Maps
Run the notebooks in the `create_maps` in the following order to create the LFMC and uncertainty maps. This assumes the [mapping models](#maps) have been created.
#### Extract modelling data from Google Earth Engine
1. `Extract Auxiliary Data Conus.ipynb`
2. `Extract MODIS Conus Image.ipynb`
3. `Extract PRISM Conus Image.ipynb`
#### Create the TIFFs used to generate maps
4. `CONUS LFMC maps.ipynb`

Maps can be created from the output TIFFs using a GIS tool such as QGIS. 

## Contributors
1. Lynn Miller: https://github.com/lynn-miller

## Acknowledgements and References

### Globe-LFMC dataset
Yebra, M., Scortechini, G., Badi, A., Beget, M.E., Boer, M.M., Bradstock, R., Chuvieco, E., Danson, F.M., Dennison, P., Resco de Dios, V., Di Bella, C.M., Forsyth, G., Frost, P., Garcia, M., Hamdi, A., He, B., Jolly, M., Kraaij, T., Martín, M.P., Mouillot, F., Newnham, G., Nolan, R.H., Pellizzaro, G., Qi, Y., Quan, X., Riaño, D., Roberts, D., Sow, M., Ustin, S., 2019. Globe-LFMC, a global plant water status database for vegetation ecophysiology and wildfire applications. Sci. Data 6, 155. https://doi.org/10.1038/s41597-019-0164-9

### Google Earth Engine Data
1. Google Earth Engine: Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., Moore, R., 2017. Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sens. Environ. 202, 18–27. https://doi.org/10.1016/j.rse.2017.06.031
2. NASA SRTM Digital Elevation 30m dataset: Farr, T.G., Rosen, P.A., Caro, E., Crippen, R., Duren, R., Hensley, S., Kobrick, M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S., Shimada, J., Umland, J., Werner, M., Oskin, M., Burbank, D., and Alsdorf, D.E. (2007). The shuttle radar topography mission: Reviews of Geophysics, v. 45, no. 2, RG2004, at https://doi.org/10.1029/2005RG000183.
3. MODIS MCD43A4 Version 6 dataset: Schaaf, C., Wang, Z. (2015). <i>MCD43A4 MODIS/Terra+Aqua BRDF/Albedo Nadir BRDF Adjusted Ref Daily L3 Global - 500m V006</i> [Data set]. NASA EOSDIS Land Processes DAAC. Accessed 2021-03-19 from https://doi.org/10.5067/MODIS/MCD43A4.006 
4. MODIS MOD10A1 Version 6 dataset: Hall, D.K., Riggs, G.A., 2016. MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid, Version 6. https://doi.org/10.5067/MODIS/MOD10A1.006
5. MODIS MOD44W Version 6 water mask: Carroll, M., DiMiceli, C., Wooten, M., Hubbard, A., Sohlberg, R., Townshend, J., 2017. MOD44W MODIS/Terra Land Water Mask Derived from MODIS and SRTM L3 Global 250m SIN Grid V006. NASA EOSDIS Land Processes DAAC. https://lpdaac.usgs.gov/products/mod44wv006
6. PRISM AN81d Gridded daily climate dataset: PRISM Climate Group, 2004. PRISM Climate Group [WWW Document]. Oregon State Univ. URL http://prism.oregonstate.edu

### Köppen-Geiger Climate Zone Data
Beck, H.E., Zimmermann, N.E., McVicar, T.R., Vergopolan, N., Berg, A., Wood, E.F., 2018. Present and future Köppen-Geiger climate classification maps at 1-km resolution. Sci. Data 5, 180214. https://doi.org/10.1038/sdata.2018.214

### TempCNN model
Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal convolutional neural network for the classification of satellite image time series. Remote Sensing, 11(5), 1–25. https://doi.org/10.3390/rs11050523

### Modis-tempCNN model
Zhu, L., Webb, G. I., Yebra, M., Scortechini, G., Miller, L., & Petitjean, F. (2021). Live fuel moisture content estimation from MODIS: A deep learning approach. ISPRS Journal of Photogrammetry and Remote Sensing, 179, 81–91. https://doi.org/10.1016/j.isprsjprs.2021.07.010

Github: Miller, L., Zhu, L., LFMC_from_MODIS [Computer software] https://github.com/lynn-miller/LFMC_from_MODIS

### Main Python Packages
1. TensorFlow: Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., … Research, G. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. www.tensorflow.org.
2. Keras (this software uses the version of Keras that comes with Tensorflow): Chollet, F., & Others. (2015). Keras. https://keras.io
3. Numpy: Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2 
4. Pandas: Reback, J., McKinney, W., jbrockmendel, Bossche, J. Van den, Augspurger, T., Cloud, P., gfyoung, Hawkins, S., Sinhrks, Roeschke, M., Klein, A., Petersen, T., Tratner, J., She, C., Ayd, W., Naveh, S., Garcia, M., patrick, Schendel, J., … h-vetinari. (2021). pandas-dev/pandas: Pandas 1.2.1. https://doi.org/10.5281/zenodo.4452601
5. Scipy: Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … Vázquez-Baeza, Y. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
6. Scikit-Learn: Pedregosa, F., Michel, V., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python
7. Multiprocess: McKerns, M., Strand, L., Sullivan, T., Fang, A., & Aivazis, M. (2011). Building a Framework for Predictive Science. Proceedings of the 10th Python in Science Conference, Scipy, 76–86. https://doi.org/10.25080/Majora-ebaa42b7-00d
8. Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. In Computing in Science & Engineering (Vol. 9, Issue 3, pp. 90–95). IEEE Computer Society. https://doi.org/10.1109/MCSE.2007.55
9. Seaborn: Waskom, M. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021
