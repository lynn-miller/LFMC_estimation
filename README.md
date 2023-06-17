# Deep Learning Models for Live Fuel Moisture Content Estimation
This repository contains code to build and evaluate deep learning models to estimate Live Fuel Moisture Content (LFMC). These models (Multi-tempCNN) use time series of reflectance and meteorological data, together with some auxiliary variables to train the CNNs to estimate LFMC.

## Branches
These branches contain the code used to create the LFMC estimation models used in publications. Code in each branch is not updated (apart from bug fixes) to ensure consistency with the paper.

Note that due to random variable seed settings and randomness in the scheduling of Tensorflow tasks on GPUs, results obtained may differ slightly from those shown in the paper, but should be broadly similar.

### Multi-modal_LFMC
Paper: Multi-modal Temporal CNNs for Live Fuel Moisture Content Estimation
Reference: Miller, L., Zhu, L., Yebra, M., Rüdiger, C., & Webb, G. I. (2022). Multi-modal temporal CNNs for live fuel moisture content estimation. Environmental Modelling & Software, 156, 105467. https://doi.org/10.1016/j.envsoft.2022.105467

### LFMC_projections
Paper: Projecting live fuel moisture content via deep learning
Reference: Miller, L., Zhu, L., Yebra, M., Rüdiger, C., & Webb, G. I. (2023). Projecting live fuel moisture content via deep learning. International Journal of Wildland Fire, 32(5), 709–727. https://doi.org/10.1071/WF22188

### Domain_adaptation
Paper: Inter-Continental Domain Adaptation for Temporal CNNs
Reference: Miller, L., Rüdiger, C., & Webb, G. I. (2023). Inter-Continental Domain Adaptation for Temporal CNNs. (Paper in progress).

## Main Directories
The python code in the top-level directory contains the modules, classes, and functions used to extract data, build models and analyse results.
 
A directory is provided that contains the notebooks and scripts used to generate and analyse the specific models used for each paper. In the "main" branch, these are updated to run with the latest version of the code. Each directory contains the customised `common.py` and `architecture_???.py` files, and has 4 sub-directories `extract_data`, `build_models`, `analyse_results` and `create_maps` that contain the scripts and notebooks for each task.

## Requirements
1. Python 3.8 with packages listed in [requirements.txt](../requirements.txt). The code was written and tested using Anaconda on Windows. It has also been tested on Linux, but there were minor differences in the versions of some packages (see [requirements-linux.txt](../requirements-linux.txt) for package versions used). [InstallNotes.md](../InstallNotes.md) contains some notes about how to set up a suitable Anaconda virtual environment.
2. An authenticated Google Earth Engine account
3. A copy of the [`Globe-LFMC.xlsx` dataset](#Globe-LFMC)
4. A copy of the [Köppen-Geiger Climate Zone Data](#Koppen-Geiger)

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
7. ERA5 dataset: Copernicus Climate Change Service (C3S), 2017. ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate [WWW Document]. Copernicus Clim. Chang. Serv. Clim. Data Store. URL https://cds.climate.copernicus.eu/cdsapp#!/home

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
10. OSGeo: GDAL/OGR contributors. (2022). GDAL/OGR Geospatial Data Abstraction software Library. https://doi.org/10.5281/zenodo.5884351
