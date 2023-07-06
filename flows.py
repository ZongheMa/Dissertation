import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import gzip
import shutil
from datetime import datetime


def merge_csv_files(directory):
    # Get a list of all the csv files
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Initialize an empty list to hold dataframes
    dfs = []

    # Loop through csv files, read each into a dataframe, and append to the list
    for file in csv_files:
        # Extract date from filename, assuming the date is in format 'traffic_flow_YYYY_MM_DD'
        date_str = file.split('.')[0].split('_')[-3:]  # This gives ['YYYY', 'MM', 'DD']
        date = datetime.strptime('_'.join(date_str), '%Y_%m_%d').date()

        df = pd.read_csv(os.path.join(directory, file))

        # Add date as a new column
        df['date'] = date.strftime('%m/%d/%y')

        dfs.append(df)

    # Concatenate all dataframes in the list into one dataframe
    merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # Return the merged dataframe
    return merged_df


# load the datasets
traffic_flows = merge_csv_files(
    '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]Traffic flow')
lsoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/london_LSOA/london_LSOA.shp'
road_network = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_network.shp'

# clean the traffic flow data
traffic_flows = traffic_flows.drop_duplicates(['toid', 'date'])
traffic_flows = traffic_flows.groupby(['toid', 'date']).agg(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'}).reset_index()

# lsoa = gpd.read_file(lsoa, crs={'init': 'epsg:27700'})
road_network = gpd.read_file(road_network, crs={'init': 'epsg:27700'})

flows = pd.merge(
    road_network[['toid', 'roadclassi', 'routehiera', 'geometry',
                  'directiona', 'length', 'roadwidthm', 'elevationg'
                  ]],
    traffic_flows, left_on='toid', right_on='toid', how='right')
# flows = gpd.GeoDataFrame(flows, crs={'init': 'epsg:27700'}, geometry='geometry')
'''
# Perform a spatial join to LSOA level
joined_gdf = gpd.sjoin(lsoa, flows, how='inner', op='contains')
result = joined_gdf.groupby(['LSOA21CD', 'date'])['bus','car','cycle','walks','stationary'].sum().reset_index()
flows_lsoa = lsoa.merge(result, left_on='LSOA21CD', right_on='LSOA21CD', how='left')

flows_lsoa.to_file('/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/processed data/flows_lsoa.shp')
'''
# Perform the aggregation to road network level

# cycle = flows.pivot(index='toid', columns='date', values='cycle')
# road_geometry = flows[['toid', 'geometry']].drop_duplicates('toid')
# cycle = cycle.merge(road_geometry, on='toid', how='left')
# gpd.GeoDataFrame(cycle, geometry='geometry')
#
# cycle.plot(column='2020-03-01', legend=True)
# plt.show()


cycle = traffic_flows.pivot(index='toid', columns='date', values='cycle')
cycle = pd.merge(road_network[['toid', 'roadclassi', 'routehiera', 'geometry',
                               'directiona', 'length', 'roadwidthm', 'elevationg']],
                 cycle,
                 left_on='toid', right_on='toid', how='left').reset_index()

# obtain the date columns
date_columns = cycle.columns[cycle.columns.str.contains('/')]

# Calculate the difference between each date column
for i, col in enumerate(date_columns[1:], 1):
    diff_col_name = f'diff_{col}'
    cycle[diff_col_name] = cycle[date_columns[i]] - cycle[date_columns[i - 1]]

# Calculate the sum and the difference weekly
week_sums = cycle[date_columns].rolling(window=7, axis=1).sum().iloc[:, 6::7]
week_sums.rename(columns={col: f'sum_week_{i}' for i, col in enumerate(week_sums.columns, 1)}, inplace=True)
cycle = pd.concat([cycle, week_sums], axis=1)
cycle['diff_week_1&2'] = cycle['sum_week_2'] - cycle['sum_week_1']
cycle['diff_week_2&3'] = cycle['sum_week_3'] - cycle['sum_week_2']

# allocate the inner and outer London


# cycle = gpd.GeoDataFrame(cycle, crs={'init': 'epsg:27700'}, geometry='geometry')
# cycle.plot(column='diff_03/01/22', legend=True, cmap='OrRd').set_axis_off()
# plt.show()
