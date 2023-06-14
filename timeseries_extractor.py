"""Time Series Extractor class"""

import ee
import numpy as np
import os
import pandas as pd


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
        date_range = pd.date_range(data.index.min(), self.end_date,
                                   freq=self.freq, inclusive="left")
        self.days = pd.Series(date_range, name="id")
        bands_df = data.merge(
            self.days, how="right", left_index=True, right_on='id').set_index('id')
        method = 'linear' if self.gap_fill == True else self.gap_fill
        bands_df = bands_df[self.bands].interpolate(
            axis=0, method=method, limit=self.max_gap, limit_direction="both")
        return bands_df


class GeeTimeseriesReduceExtractor(GeeTimeseriesExtractor):
    """Google Earth Engine Time Series Extractor with Reducer class
    
    GEE Time Series extractor for image collections with more than one
    image per daily (e.g. hourly images). Reduces the images to daily
    images.
    
    NOTE: reducers act on all bands. The band parameter is used to
    select the band/reducer combinations required.
    
    Parameters
    ----------
    product : str
        The Google Earth Engine product name.
    bands : dict
        keys: final band names
        values: reduced band names (<collection band name>_<reducer>).
    reducers : list
        The list of reducers to apply.
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

    def __init__(self, product, bands, reducers, start_date, end_date,
                 freq='1D', gap_fill=True, max_gap=None, dir_name=''):
        self.product = product
        self.bands = bands
        self.reducers = reducers
        self.collection = ee.ImageCollection(product)
        self.set_date_range(start_date, end_date, freq, gap_fill, max_gap)
        self.save_band_info()
        self.set_output_dir(dir_name)
        self.set_default_proj_scale()
        
    def extract_data(self, location):

        def reduce_daily(day_offset):
            start = gee_start.advance(day_offset, 'day')
            end = start.advance(1, 'day')
            reducers = self.reducers[0]
            for reducer in self.reducers[1:]:
                reducers = reducers.combine(reducer, sharedInputs=True)
            coll = self.collection.filterDate(start, end).reduce(reducers)
            coll = coll.set('system:time_start', start.millis())
            return coll.select(list(self.bands.values()), list(self.bands.keys()))

        geometry = ee.Geometry.Point([location.Longitude, location.Latitude])
        gee_start = ee.Date(self.start_date)
        gee_end = ee.Date(self.end_date)
        gee_days = gee_end.difference(gee_start, 'days').getInfo()
        data = ee.ImageCollection(ee.List.sequence(0, gee_days-1).map(reduce_daily))
        data = data.getRegion(geometry, self.scale, self.projection).getInfo()
        data_df = pd.DataFrame(data[1:], columns=data[0])
        bands_index = pd.DatetimeIndex(pd.to_datetime(data_df.time, unit='ms').dt.date)
        bands_df = data_df[list(self.bands.keys())]
        bands_df = bands_df.set_index(bands_index).rename_axis(index='id').sort_index()
        return bands_df

