"""Data extraction utilities."""

import numpy as np
import pandas as pd
import re
from datetime import datetime
from datetime import timedelta
from osgeo import gdal

from data_prep_utils import normalise


def get_sample_data(date_str, site_data, ts_offset=1, ts_length=365, ts_freq=1):
    """Retrieves a timeseries for a sample from the full site data.
    
    Parameters
    ----------
    date_str : str
        The sampling date in YYYY-MM-DD format.
    site_data : DataFrame
        The full data extract for a site. Columns are the channels or
        bands of the source data. The dataframe should have a datetime
        index with 1 row per day.
    ts_offset : int, optional
        The number of days before the sampling date the time series
        should end. The default is 1 - the time series will end the day
        before the sampling date.
    ts_length : int, optional
        The length of the timeseries. The number of steps in the
        timeseries (not the number of days spanned by the timeseries).
        The default is 365.
    ts_freq : int, optional
        The frequency (in days) of the time series data. The default is
        1 - a daily time series.

    Returns
    -------
    np.array or None
        The flattened timeseries, in date then band order. ``None`` if
        the timeseries cannot be extracted from ``site_data``.
    """
    sample_date = datetime.strptime(date_str, '%Y-%m-%d')
    end_date = sample_date - timedelta(days=ts_offset)
    date_range = pd.date_range(end=end_date, periods=ts_length, freq=timedelta(days=ts_freq))
    try:
        return site_data.loc[date_range].values.flatten()
    except:
        print(f'Invalid date: {date_str} outside valid date range')
        return None


def extract_timeseries_data(extractor, sites, samples, earliest_sample_date,
                            ts_offset=1, ts_length=365, ts_freq=1):
    """Extracts time series data for a site
    
    Extracts time series data for all samples at a set of sites using a
    TimeseriesExtractor. Reports (prints) invalid samples and sites -
    those for which data cannot be extract (as well as returning a list
    of these).

    Parameters
    ----------
    extractor : TimeseriesExtractor
        The TimeseriesExtractor to extract data from.
    sites : DataFrame
        The sites to extract data for. Must have "Site", "Longitude" and
        "Latitude" columns.
    samples : DataFrame
        The samples to extract data for. Must have "ID" and
        "Sampling date" columns.
    earliest_sample_date : datetime
        The earliest date a sample can have to be valid. Samples with
        earlier dates will be returned as invalid_samples, but will not
        be otherwise reported
    tsOffset : int, optional
        The number of days prior to the sample's sampling date that the
        time series should end. The default is 1.
    tsLength : int, optional
        The length of the time series. The default is 365.
    tsFreq : int, optional
        The interval (in days) between timeseries entries. The default
        is 1.

    Returns
    -------
    ts_data : DataFrame
        The extracted timeseries data. Columns are the sample ID, then
        the stacked bands in date then bands order. There is a row for
        each valid sample.
    valid_data : list of bool
        A list with an entry for each sample indicating if data for the
        sample was successful extracted.
    invalid_samples : list of str
        A list of sample IDs for which data could not be extracted.
    invalid_sites : list of str
        A list of sites for which data could not be extracted.
    """
    ts_data = []
    valid_data = [False] * samples.shape[0]
    invalid_samples = []
    invalid_sites = []
    for site_idx, site in sites.iterrows():
        print(f'Processing site {site.Site}')
        site_samples = samples[samples.Site == site.Site]
        try:
            site_data = extractor.get_and_save_data(site)
        except:
            print(f'Failed to extract data for {site.Site}')
            invalid_sites.append(site.Site)
            continue
        if site_data.notna().sum().sum() == 0:
            print(f'No data found for {site.Site}')
            invalid_sites.append(site.Site)
            continue
        for index, sample in site_samples.iterrows():
            # Only process samples on or after the earliest sampling date -
            # Discard samples before this date - but don't bother reporting them
            if datetime.strptime(sample["Sampling date"], '%Y-%m-%d') >= earliest_sample_date:
                sample_data = get_sample_data(sample["Sampling date"], site_data,
                                              ts_offset, ts_length, ts_freq)
                if sample_data is None or np.isnan(sample_data.sum()):
                    invalid_samples.append(index)
                else:
                    ts_data.append([sample.ID] + list(sample_data))
                    valid_data[index] = True
    return ts_data, valid_data, invalid_samples, invalid_sites


def sort_key(key):
    """Returns a numeric representation of a site or sample key
    
    Generates an integer key from a site or sample key. Assumes the key
    is a character followed by numbers between 1 and 999 separated by
    underscores. The leading character is ignored and the numeric parts
    processed from left to right.
    
    Example:
      - key: C13_1_4; generated key: 13001004
      - key: C6_1_13; generated key: 6001013
      - key: C6_1_5; generated key: 6001005
     
    These will be ordered as C6_1_5, C6_1_13, and C13_1_4 when sorted
    by the ``sortKey()`` value, a more natural ordering than the
    original alphanumeric sorting.
    
    Parameters
    ----------
    key : str
        Either a site (e.g. C6_1) or sample key (e.g. C6_1_13).

    Returns
    -------
    genKey : int
        A numeric representation of the key that can be used to sort
        keys into logical order.
    """
    key_parts = key.split("_")
    genKey = int(re.findall("\d+", key_parts[0])[0])
    for key_part in key_parts[1:]:
        genKey = genKey * 1000 + int(key_part)
    return genKey


def extract_koppen_data(koppen_file, legend_file, data,
                        loc_columns=['longitude', 'latitude'],
                        cz_columns=['Czone1', 'Czone2', 'Czone3'],
                        nodata_value='',
                        append_to_data=True):
    """ Extracts climate zone data
    
    Extract climate zone data from a raster file

    Parameters
    ----------
    koppen_file : str
        The name of the climate zone raster file. The file can be any
        format GDAL can read.
    legend_file : str
        The name of the file containing the climate zone codes. The
        file should be a csv file with headers, with the numeric codes
        in the first column and the alphabetic codes in the ``Code``
        column.
    data : pd.DataFrame
        Data Frame of sites.
    loc_columns : list, optional
        Column names of columns containing the site longitude and
        latitude. The default is ['longitude', 'latitude'].
    cz_columns : list, optional
        Names for the three climate zone columns. The default is
        ['Czone1', 'Czone2', 'Czone3'].
    nodata_value : str, optional
        Value to use if climate zone is invalid. The default is ''.
    append_to_data : bool, optional
        If True, climate zones are added to the ``data`` dataframe.
        If False, a list of the climate zones is returned. Has an entry
        for each row in ``data`` (site), which is a list of the three
        climate zone levels for the site. The default is True.

    Returns
    -------
    kop_values : list
        If ``append_to_data`` is False, a list of the climate zones. It
        has an entry for each row in ``data`` (site), which is a list
        of the three climate zone levels for the site. 
    """
    legend = pd.read_csv(legend_file, index_col=0)
    kop_data = gdal.Open(koppen_file)
    transform = kop_data.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    kop_values = [[], [], []]
    invalid_count = {}
    for idx, site in data.iterrows():
        x_offset = int((site[loc_columns[0]] - x_origin) / pixel_width)
        y_offset = int((site[loc_columns[1]] - y_origin) / pixel_height)
        if (x_offset < 0 or x_offset > kop_data.RasterXSize
            or y_offset < 0 or y_offset > kop_data.RasterYSize):
            print(f'Out of range: {site[loc_columns[0]]}, {site[loc_columns[1]]}')
        kop = kop_data.ReadAsArray(x_offset, y_offset, 1, 1)
        if kop[0][0] in [31, 32]:
            # Fix for invalid raster values - 31 (ETH) should be 29 (ET) and 32 (EFH) should
            # be 30 (EF). See Peel et al. (2007) - DOI 10.5194/hess-11-1633-2007 for details
            kop[0][0] -= 2
        try:
            kop = legend.Code[kop[0][0]]
            kop_codes = [kop[:1], kop[:2], kop]
        except:
            invalid_count[kop[0][0]] = invalid_count.get(kop[0][0], 0) + 1
            kop_codes = [nodata_value, nodata_value, nodata_value] 
        for i, c in enumerate(kop_codes):
            kop_values[i].append(c)
    for code, num in invalid_count.items():
        print(f'Code {code}: {num} record{"" if num == 1 else "s"}')
    if append_to_data:
        for i, kop in enumerate(kop_values):
            data[cz_columns[i]] = kop
    else:
        return kop_values
    

def normalise_dem(data,
                  input_columns=['longitude', 'latitude', 'elevation', 'slope', 'aspect'],
                  precision=5):
    """Normalises DEM data
    
    Normalise the location and DEM data.

    Parameters
    ----------
    data : DataFrame
        A dataframe containing the data to be normalised.
    input_columns : list, optional
        The columns in ``data`` that represent the longitude, latitude,
        elevation, slope and aspect. The default is
        ``['longitude', 'latitude', 'elevation', 'slope', 'aspect']``.
    precision : int, optional
        The precision of the normalised data. The default is 5.

    Returns
    -------
    dem_norm : TYPE
        DESCRIPTION.

    """
    dem_norm = data[input_columns].round(precision)
    longitude = normalise(
        data[input_columns[0]], method='range', data_range=(-180, 180),
        scaled_range=(-np.pi, np.pi))
    dem_norm["Long_sin"] = longitude.transform(np.sin).round(precision)
    dem_norm["Long_cos"] = longitude.transform(np.cos).round(precision)
    dem_norm["Lat_norm"] = normalise(
        data[input_columns[1]], method='range', data_range=(-90, 90)).round(precision)
    dem_norm["Elevation"] = normalise(
        data[input_columns[2]], method='range', data_range=(0, 6000)).round(precision)
    dem_norm["Slope"] = normalise(
        data[input_columns[3]], method='range', data_range=(0, 90)).round(precision)
    aspect = normalise(
        data[input_columns[4]], method='range', data_range=(0, 360),
        scaled_range=(-np.pi, np.pi))
    dem_norm["Aspect_sin"] = aspect.transform(np.sin).round(precision)
    dem_norm["Aspect_cos"] = aspect.transform(np.cos).round(precision)
    return dem_norm