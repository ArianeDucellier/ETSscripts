"""
This module allows you to plot tremor data interactively
with the Python package altair
"""

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon

def select_tremor(tremors, tbegin, tend, \
    latmin, latmax, lonmin, lonmax):
    """
    Select tremor within user-defined area and time range

    Input:
        type tremors = pandas dataframe
        tremors = {datetime, latitude, longitude, depth}
        type tbegin = datatime.datetime
        tbegin = Beginning of selected time interval
        type tend = datatime.datetime
        tend = End of selected time interval
        type latmin = float
        latmin = Southern boundary of selected region
        type latmax = float
        latmax = Northern boundary of selected region
        type latmin = float
        lonmin = Western boundary of selected region
        type latmin = float
        lonmax = Eastern boundary of selected region
    Output:
        type tremors = pandas dataframe
        tremors = {datetime, latitude, longitude, depth}
    """
    # Keep only tremors within a user-defined area
    if (latmin != None):
        tremors = tremors.loc[(tremors['latitude'] >= latmin)]
    if (latmax != None):
        tremors = tremors.loc[(tremors['latitude'] <= latmax)]
    if (lonmin != None):
        tremors = tremors.loc[(tremors['longitude'] >= lonmin)]
    if (lonmax != None):
        tremors = tremors.loc[(tremors['longitude'] <= lonmax)]
    # Keep only tremors within a user-defined time range
    if (tbegin !=None):
        mask = (tremors['datetime'] >= tbegin)
        tremors = tremors.loc[mask]
    if (tend != None):
        mask = (tremors['datetime'] <= tend)
        tremors = tremors.loc[mask]
    return tremors

def bin_tremor(tremors, nbin, winlen):
    """
    Compute the percentage of the time during which there is recorded tremor

    Input:
        type tremors = pandas dataframe
        tremors = {datetime, latitude, longitude, depth}
        type nbin = integer
        nbin = Duration of the time windows (in minutes) for which we compute
            the percentage of time with tremor
        type winlen = float
        winlen = Duration of the time windows from the tremor catalog
            (in minutes)
    Output:
        type dfInterp = pandas dataframe
        dfInterp = {datetime, latitude, longitude, depth, Time, Value}
    """
    # Bin tremor windows
    smin = str(nbin) + 'T'
    df = pd.DataFrame({'Time': tremors['datetime'], \
                       'Value': np.repeat(1, tremors.shape[0])})
    df.set_index('Time', inplace=True)
    df_group = df.groupby(pd.Grouper(level='Time', \
        freq=smin))['Value'].agg('sum')   
    df_group = df_group.to_frame().reset_index()
    df_group['Value'] = (winlen / nbin) * df_group['Value']
    # Merge datasets to keep the number of tremor windows
    dfInterp = pd.merge_asof(tremors.sort_values(by="datetime"), \
        df_group.sort_values(by="Time"), left_on="datetime", right_on="Time")
    return dfInterp

def read_background(filename, latmin, latmax, lonmin, lonmax):
    """
    Get a geodataframe with a background map

    Input:
        type filename = string
        filename = Shape file with background map
        type latmin = float
        latmin = Southern boundary of the map
        type latmax = float
        latmax = Northern boundary of the map
        type lonmin = float
        lonmin = Western boundary of the map
        type lonmax = float
        lonmax = Eastern boundary of the map
    Output:
        type subdata = geopandas dataframe
        subdata = background map
    """
    data = gpd.read_file(filename)
    data.rename(columns={'id' : 'id0'}, inplace=True)
    limits = gpd.GeoSeries([Polygon([(lonmin, latmin), (lonmax, latmin), \
        (lonmax, latmax), (lonmin, latmax)])])
    boundaries = gpd.GeoDataFrame({'geometry': limits, 'df':[1]}, crs=data.crs)
    subdata = gpd.overlay(data, boundaries, how='intersection')
    return subdata

def plot_tremor(tremors):
    """
    Plot tremor location and tremor activity
    with interaction between both graphs

    Input:
        type tremors = pandas dataframe
        tremors = {datetime, latitude, longitude, depth, Time, Value}
    Output:
        type myChart = Altair chart
        myChart = tremor plot
    """
    # Selection
    brush = alt.selection(type='interval', encodings=['x'])
    # Map of tremor location
    points = alt.Chart(
    ).mark_point(
    ).encode(
        longitude = 'longitude',
        latitude = 'latitude',
        color=alt.Color('Time', \
                        legend=alt.Legend(format='%Y/%m/%d - %H:%M:%S'))
    ).transform_filter(
        brush.ref()
    ).properties(
        width=600,
        height=600
    )
    # Graph of tremor activity
    bars = alt.Chart(
    ).mark_area(
    ).encode(
        x=alt.X('Time', \
                axis=alt.Axis(format='%Y/%m/%d - %H:%M:%S', title='Time')),
        y=alt.Y('Value', \
                axis=alt.Axis(format='%', title='Percentage of tremor'))
    ).properties(
        width=600,
        height=100,
        selection=brush
    )
    # Putting graphs together
    myChart = alt.vconcat(points, bars, data=tremors)
    return myChart

def plot_tremor_withbg(tremors, subdata):
    """
    Plot tremor location and tremor activity
    with interaction between both graphs and a background map

    Input:
        type tremors = pandas dataframe
        tremors = {datetime, latitude, longitude, depth, Time, Value}
        type subdata = geopandas dataframe
        subdata = Background map
    Output:
        type myChart = Altair chart
        myChart = tremor plot
    """
    # Selection
    brush = alt.selection(type='interval', encodings=['x'])
    # Background map
    bgmap = alt.Chart(subdata).mark_geoshape(
    ).project(
    ).encode(
        color=alt.value('white')
    ).properties(
    )
    # Map of tremor location
    points = alt.Chart(tremors
    ).mark_point(
    ).encode(
        longitude = 'longitude',
        latitude = 'latitude',
        color=alt.Color('Time', \
                        legend=alt.Legend(format='%Y/%m/%d - %H:%M:%S'))
    )
    # Add background map to tremor location
    bgpoints = alt.layer(bgmap, points
    ).transform_filter(
        brush.ref()
    ).properties(
        width=600,
        height=600
    )
    # Graph of tremor activity
    bars = alt.Chart(tremors
    ).mark_area(
    ).encode(
        x=alt.X('Time', \
                axis=alt.Axis(format='%Y/%m/%d - %H:%M:%S', title='Time')),
        y=alt.Y('Value', \
                axis=alt.Axis(format='%', title='Percentage of tremor'))
    ).properties(
        width=600,
        height=100,
        selection=brush
    )
    # Putting graphs together
    myChart = alt.vconcat(bgpoints, bars).configure(background='lightblue')
    return myChart

def visualize_tremor(filename, output, nbin, bg=False, winlen=1.0, \
    tbegin=None, tend=None, \
    latmin=None, latmax=None, lonmin=None, lonmax=None):
    """
    Read and plot tremor location and activity

    Input:
        type filename = string
        filename = Pickle file where tremor dataset is stored
        type output = string
        output = Name of output file containing the figure
        type bg = boolean
        bg = Do we add a background map?
        type nbin = integer
        nbin = Duration of the time windows (in minutes) for which we compute
            the percentage of time with tremor
        type winlen = float
        winlen = Duration of the time windows from the tremor catalog
            (in minutes)
        type tbegin = datatime.datetime
        tbegin = Beginning of selected time interval
        type tend = datatime.datetime
        tend = End of selected time interval
        type latmin = float
        latmin = Southern boundary of selected region
        type latmax = float
        latmax = Northern boundary of selected region
        type latmin = float
        lonmin = Western boundary of selected region
        type latmin = float
        lonmax = Eastern boundary of selected region
    Output:
        None
    """
    # Read dataset
    tremors = pd.read_pickle(filename)[0]
    # Select tremors
    tremors = select_tremor(tremors, tbegin, tend, \
        latmin, latmax, lonmin, lonmax)
    # Construct time line for selection
    tremors = bin_tremor(tremors, nbin, winlen)
    # Manage big datasets
    alt.data_transformers.enable('json')
    # Background map
    if (bg == True):
        subdata = read_background('data/WA_BC_Plus_shoreline.shp', \
            tremors['latitude'].min(), tremors['latitude'].max(), \
            tremors['longitude'].min(), tremors['longitude'].max())
        myChart = plot_tremor_withbg(tremors, subdata)
    else:        
        # Plot
        myChart = plot_tremor(tremors)
    # Save
    myChart.save(output + '.html')

if __name__ == '__main__':

    filename = 'data/ghosh_al_2012_JGR.pkl'
#    filename = 'data/PNSN_2009-2018_NW.pkl'
    output = 'ghosh_al_2012_JGR_withbg'
#    output = 'PNSN_2009-2018_NW'
    winlen = 1.0
#    winlen = 2.5
    nbin =  1440
    bg = True
    visualize_tremor(filename, output, nbin, bg, winlen, \
        tbegin=None, tend=None, \
        latmin=None, latmax=None, lonmin=None, lonmax=None)
