"""Time Series Extractor class"""

import ee
import glob
import numpy as np
import os
import pandas as pd
import re
import time
from osgeo import gdal


class TimeseriesExtractor:
    """Time Series Extractor class
    
    A class for a timeseries extractor. This is a abstract class.
    Subclasses need to implement methods:
     -  __init__: The method here shows the minimum needed.
     -  save_band_info: should set any required information about the
        bands to the band_info attribute and the datatype of the band
        data to the datatype attribute.
     -  extract_data: should return a dataframe. The columns are the
        bands and rows are the time points. It should be indexed using
        a Pandas Datetime index and gap-filled if necessary
    
    Parameters
    ----------
    source : str or open GDAL file
        The source of the timeseries data.
    bands : list
        A list of the band names required.
    start_date : str
        The start date for the time series.
    end_date : str
        The end date for the time series.
    freq : str, optional
        The frequency of time series entries. The default is '1D' for
        daily entries.
    gap_fill : str or bool, optional
        Indicates if the extracted timeseries should be gap-filled. The
        default is True.
    max_gap : int, optional
        The maximum size of gap that will be filled. Filling is done in
        both directions, so the maximum actual size filled is
        ``2 * max_gap``. If None, there is no limit on the gap size.
        The default is None.
    dir_name : str
        The directory where the extracted files will be stored. The
        default is ''.

    Returns
    -------
    None.
    """

    _int_data_types = ['Int8', 'UInt8', 'Int16', 'UInt16', 'Int32', 'UInt32', 'Int64', 'UInt64']
    
    def __init__(self, source, bands, start_date, end_date,
                 freq='1D', gap_fill=True, max_gap=None, dir_name=''):
        self.source = source
        self.bands = bands
        self.set_date_range(start_date, end_date, freq, gap_fill, max_gap)
        self.save_band_info()
        self.set_output_dir(dir_name)
         
    def _int_data(self):
        return self.data_type in self._int_data_types
        
    def set_date_range(self, start_date, end_date, freq='1D', gap_fill=True, max_gap=None):
        self.start_date = start_date
        self.end_date = end_date
        if gap_fill:
            date_range = pd.date_range(start_date, end_date, freq=freq)
            self.days = pd.Series(date_range, name="id")
            self.freq = freq
            self.gap_fill = gap_fill
            self.max_gap = max_gap
        else:
            self.gap_fill = False

    def set_output_dir(self, dir_name):
        if dir_name is None or dir_name == '':
            self.dir_name = ''
        else:
            self.dir_name = os.path.join(dir_name, '')
    
    def save_band_info(self, location):
        self.data_type = None
        raise NotImplementedError
    
    def extract_data(self, location):
        raise NotImplementedError
    
    def gap_fill_data(self, data):
        raise NotImplementedError
            
    def get_and_save_data(self, location):
        file_name = f'{self.dir_name}{location.Site}.csv'
        try:  # If we already have the location data, read it
            point_df = pd.read_csv(file_name, index_col="id", parse_dates=True)
        except:  # Otherwise extract it from GEE
            print(f'Extracting data for {location.Site} '
                  + f'(lat: {location.Latitude} long: {location.Longitude})')
            point_df = self.extract_data(location)
            point_df.to_csv(file_name)
        if self.gap_fill:
            point_df = self.gap_fill_data(point_df)
        if self._int_data():
            point_df = point_df.round().astype(self.data_type)
        return point_df
    

class GeeTimeseriesExtractor(TimeseriesExtractor):
    """Google Earth Engine Time Series Extractor class
    
    Parameters
    ----------
    product : str
        The Google Earth Engine product name.
    bands : list
        A list of the band names required.
    start_date : str
        The start date for the time series.
    end_date : str
        The end date for the time series.
    freq : str, optional
        The frequency of time series entries. The default is '1D' for
        daily entries.
    gap_fill : str or bool, optional
        Indicates if the extracted timeseries should be gap-filled. The
        default is True.
    max_gap : int, optional
        The maximum size of gap that will be filled. Filling is done in
        both directions, so the maximum actual size filled is
        ``2 * max_gap``. If None, there if no limit on the gap size.
        The default is None.
    dir_name : str
        The directory where the extracted files will be stored. The
        default is ''.

    Returns
    -------
    None.
    """

    def __init__(self, product, bands, start_date, end_date,
                 freq='1D', gap_fill=True, max_gap=None, dir_name=''):
        self.product = product
        self.bands = bands
        self.collection = ee.ImageCollection(product).select(bands)
        self.set_date_range(start_date, end_date, freq, gap_fill, max_gap)
        self.save_band_info()
        self.set_output_dir(dir_name)
        self.set_default_proj_scale()
        
    def filtered_collection(self):
        return self.collection.filterDate(self.start_date, self.end_date)
        
    def save_band_info(self):
        image_info = self.filtered_collection().first().getInfo()
        self.band_info = image_info['bands']
        data_type = self.band_info[0]['data_type']  # Assumes all bands have the same datatype
        self.data_type = 'Int64' if data_type['precision'] == 'int' else np.float
        
    def set_default_proj_scale(self):
        band = self.band_info[0]  # Assumes all bands have the same proj/scale
        self.projection = band['crs']
        self.scale = abs(band['crs_transform'][0]) # crs_tranform is [+/-scale, 0, x, 0 , +/-scale, y]
        
    def set_proj_scale(self, proj, scale):
        self.projection = proj
        self.scale = scale
        
    def extract_data(self, location):
        geometry = ee.Geometry.Point([location.Longitude, location.Latitude])
        data = self.filtered_collection().getRegion(geometry, self.scale, self.projection).getInfo()
        data_df = pd.DataFrame(data[1:], columns=data[0])
        bands_index = pd.DatetimeIndex(pd.to_datetime(data_df.time, unit='ms').dt.date)
        bands_df = data_df[self.bands].set_index(bands_index).rename_axis(index='id').sort_index()
        return bands_df
    
    def gap_fill_data(self, data):
        date_range = pd.date_range(data.index.min(), self.end_date, freq=self.freq, closed="left")
        self.days = pd.Series(date_range, name="id")
        bands_df = data.merge(
            self.days, how="right", left_index=True, right_on='id').set_index('id')
        method = 'linear' if self.gap_fill == True else self.gap_fill
        bands_df = bands_df[self.bands].interpolate(
            axis=0, method=method, limit=self.max_gap, limit_direction="both")
        return bands_df


class TifTimeseriesExtractor(TimeseriesExtractor):
    """TIFF Time Series Extractor class
    
    Extracts a time series of values from a single TIFF that has the
    time series as stacked bands (in date/time, then band order).
    
    Parameters
    ----------
    tif : str
        Full path name of the TIFF.
    bands : list
        A list of the band names.
    start_date : str
        The start date for the time series.
    end_date : str
        The end date for the time series.
    freq : str, optional
        The frequency of time series entries. The default is '1D' for
        daily entries.
    gap_fill : str or bool, optional
        Indicates if the extracted timeseries should be gap-filled. The
        default is True.
    max_gap : int, optional
        The maximum size of gap that will be filled. Filling is done in
        both directions, so the maximum actual size filled is
        ``2 * max_gap``. If None, there if no limit on the gap size.
        The default is None.
    dir_name : str
        The directory where the extracted files will be stored. The
        default is ''.

    Returns
    -------
    None.
    """

    def __init__(self, tif, bands, tif_dates, nodata_value=0.0,
                 freq='1D', gap_fill=True, max_gap=None, dir_name=''):
        self.tif = gdal.Open(tif, gdal.GA_ReadOnly)
        self.bands = bands
        self.num_bands = len(bands)
        self.date_index = pd.DatetimeIndex(tif_dates)
        self.nodata_value = nodata_value
        self.set_date_range(tif_dates[0], tif_dates[-1], freq, gap_fill, max_gap)
        self.save_band_info()
        self.save_tif_info()
        self.set_output_dir(dir_name)
        
    def save_band_info(self):
        data_type = self.tif.GetRasterBand(1).DataType   # Assumes all bands have the same datatype
        # GDAL integer types are 1 (Byte), 2 (UInt16), 3 (Int16), 4 (UInt32), 5 (Int32)
        self.data_type = 'Int8' if data_type == 1 else np.float if data_type > 5 else data_type
         
    def save_tif_info(self):
        transform = self.tif.GetGeoTransform()
        self.x_origin = transform[0]
        self.y_origin = transform[3]
        self.pixel_width = transform[1]
        self.pixel_height = transform[5]

    def extract_data(self, location):
        x_offset = int((location.Longitude - self.x_origin) / self.pixel_width)
        y_offset = int((location.Latitude - self.y_origin) / self.pixel_height)
        if (x_offset < 0 or y_offset < 0 or x_offset > self.tif.RasterXSize
                or y_offset > self.tif.RasterYSize):
            return None
        all_bands = self.tif.ReadAsArray(x_offset, y_offset, 1, 1)
        bands_df = pd.DataFrame(
            all_bands.reshape(self.tif.RasterCount//self.num_bands, self.num_bands),
            index=self.date_index)
        bands_df.columns = self.bands
        if self.gap_fill:
            bands_df = bands_df.merge(
                self.days, how="right", left_index=True, right_on='id').set_index('id')
            method = 'linear' if self.gap_fill == True else self.gap_fill
            bands_df = bands_df.mask(bands_df == self.nodata_value).interpolate(
                axis=0, method=method, limit=self.max_gap, limit_direction="both")
        if self._int_data():
            bands_df = bands_df.round().astype(self.data_type)
        return bands_df


class NetcdfTimeseriesExtractor(TimeseriesExtractor):
    """NetCDF Time Series Extractor class
    
    Extracts a time series of values from a set of NetCDF files.
    
    Parameters
    ----------
    netcdf_dir : str
        Full path name of the TIFF.
    bands : list
        A list of the band names.
    start_date : str
        The start date for the time series.
    end_date : str
        The end date for the time series.
    freq : str, optional
        The frequency of time series entries. The default is '1D' for
        daily entries.
    gap_fill : str or bool, optional
        Indicates if the extracted timeseries should be gap-filled. The
        default is True.
    max_gap : int, optional
        The maximum size of gap that will be filled. Filling is done in
        both directions, so the maximum actual size filled is
        ``2 * max_gap``. If None, there if no limit on the gap size.
        The default is None.
    dir_name : str
        The directory where the extracted files will be stored. The
        default is ''.

    Returns
    -------
    None.
    """

    def __init__(self, netcdf_dir, bands, start_date, end_date, nodata_value=-9999.0,
                 netcdf_files="*.nc", freq='1D', gap_fill=True, max_gap=None, dir_name=''):
        self.netcdf_dir = netcdf_dir
        self.netcdf_files = netcdf_files
        self.bands = bands
        self.num_bands = len(bands)
        self.nodata_value = nodata_value
        self.set_date_range(start_date, end_date, freq, gap_fill, max_gap)
        self.set_output_dir(dir_name)
        
    def save_band_info(self):
        # Get the data type for the first site - assumes all sites are the same
        for site, site_data in self.data.items():
            self.data_type = site_data.dtype
            break

    def extract_data(self, location):
        all_bands = self.data[location.Site]
        bands_df = pd.DataFrame(all_bands, index=self.date_index)
        bands_df.columns = self.bands
        if self.gap_fill:
            bands_df = bands_df.merge(
                self.days, how='right', left_index=True, right_on='id').set_index('id')
            method = 'linear' if self.gap_fill == True else self.gap_fill
            bands_df = bands_df.mask(bands_df == self.nodata_value).interpolate(
                axis=0, method=method, limit=self.max_gap, limit_direction='both')
        if self._int_data():
            bands_df = bands_df.round().astype(self.data_type)
        return bands_df

    def get_date_re(self, date_format):
        if date_format in ['%Y%m%d', '%d%m%Y']:
            # This is good enough assuming the file names don't have other long strings of numbers
            return r'\d{8}'
        else:
            # Unknown date_format; replace known date components
            return date_format.replace('%Y', r'\d{4}').replace(
                '%m', r'\d{2}').replace('%d', r'\d{2}')
    
    def load_netcdf_data(self, sites, date_format='%Y%m%d', date_expr=None):
        date_expr = date_expr or self.get_date_re(date_format)
        start_time = time.time()
        all_files = glob.glob(os.path.join(self.netcdf_dir, '**', self.netcdf_files))
        all_files.sort()
        file_list = []
        date_list = []
        # Get the files for the required dates
        for file in all_files:
            file_date = re.search(date_expr, file).group()
            if pd.to_datetime(file_date, format=date_format) in self.days.array:
                file_list.append(file)
                date_list.append(file_date)
        # Extract the data for each site
        self.data = {site: np.zeros((len(file_list), len(self.bands))) for site in sites.Site}
        for n1, file in enumerate(file_list):
            for n2, band in enumerate(self.bands):
                gdal_file = gdal.Open(f'NETCDF:"{file}":{band}', gdal.GA_ReadOnly)
                transform = gdal_file.GetGeoTransform()
                for _, site in sites.iterrows():
                    x_offset = int((site.Longitude - transform[0]) / transform[1])
                    y_offset = int((site.Latitude - transform[3]) / transform[5])
                    site_data = self.data[site.Site]
                    site_data[n1][n2] = gdal_file.ReadAsArray(x_offset, y_offset, 1, 1)[0][0]
        print(round(time.time() - start_time, 2))
        self.date_index = pd.DatetimeIndex(pd.to_datetime(date_list, format=date_format))
        self.save_band_info()
