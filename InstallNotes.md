# Install Notes
These notes should aid installation of the pre-requisite Python packages into an Anaconda virtual environment on Windows or Linux.

## Pre-requisites
Assumes Anaconda has been installed. On Windows, Visual basic is also needed for CUDA

## Create a python virtual environment
```
conda create --name LFMC python=3.8
conda activate LFMC
```

## Install Conda and Python packages
```
conda install pandas ipykernel jupyter matplotlib xlrd openpyxl scikit-learn scipy pydot seaborn
conda install -c conda-forge multiprocess
conda install -c conda-forge gdal
conda install -c anaconda cudatoolkit=10.1
conda install -c anaconda cudnn=7.6.5=cuda10.1_0
```

### On Windows, install Tensorflow using pip
```
pip install tensorflow==2.3
```

### On Unix, install Tensorflow-GPU using conda
```
conda install tensorflow-gpu
```

### Notes
- Gdal needs to be installed from conda-forge to get bigtiff support
- There appear to be incompatibilities between gdal and tensorflow. Installing gdal before tensorflow seems to resolve most issues - but if using both in one Python script, you need to import tensorflow before gdal. This needs to be done even if the gdal and/or tensorflow import is done in an imported module.

## Check tensorflow works
In a python console run:
```
import tensorflow as tf
tf.__version__

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
```

## Install and authenticate Google Earth Engine
```
conda install -c conda-forge earthengine-api
earthengine authenticate
```
- This opens a browser window that requires you to sign in to google and then displays a verification code that needs to be copied and pasted into the command window at the prompt.

## Add the virtual environment to Jupyter, if not already there:
- `python -m ipykernel install --user --name=LFMC`

## Install JupyterLab and Spyder using Anaconda navigator
- Start menu shortcuts should be created automatically for Jupyter and Spyder, but change them to have the correct root directory - change the "%USERPROFILE%/" at the end of the Target to "<required directory>/"
- To create a shortcut for JupyterLab, copy and rename the Jupyter shortcut, and change "jupyter-notebook-script.py" to "jupyter-lab-script.py" in the Target

## Links
The instructions here are fairly complete and easy to follow:
https://towardsdatascience.com/setting-up-tensorflow-gpu-with-cuda-and-anaconda-onwindows-2ee9c39b5c44

This article has some more info and some code to test the install:
https://yann-leguilly.gitlab.io/post/2019-10-08-tensorflow-and-cuda/


