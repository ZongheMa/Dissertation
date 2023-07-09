import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import gzip
import shutil
from datetime import datetime
import matplotlib.colors as mcolors


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


# return the top 25% of absolute values
def top_10_abs(group, col):
    # sort the group by the absolute value of the column
    sorted_group = group.sort_values(by=col, key=abs, ascending=False)
    # get the top 25% of the group
    n = max(1, int(len(sorted_group) * 0.1))
    return sorted_group.iloc[:n]


# load the datasets
traffic_flows = merge_csv_files(
    '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]Traffic flow')
lsoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/london_LSOA/london_LSOA.shp'
road_network = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_network.shp'
inoutter = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/lp-consultation-oct-2009-inner-outer-london-shp/lp-consultation-oct-2009-inner-outer-london.shp'

# clean the traffic flow data
traffic_flows = traffic_flows.drop_duplicates(['toid', 'date'])
traffic_flows = traffic_flows.groupby(['toid', 'date']).agg(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'}).reset_index()

# lsoa = gpd.read_file(lsoa, crs={'init': 'epsg:27700'})
road_network = gpd.read_file(road_network, crs={'init': 'epsg:27700'})
inoutter = gpd.read_file(inoutter)
inoutter.to_crs(epsg=27700, inplace=True)

flows = pd.merge(
    road_network[['toid', 'roadclassi', 'routehiera', 'geometry',
                  'directiona', 'length', 'roadwidthm', 'elevationg'
                  ]],
    traffic_flows, left_on='toid', right_on='toid', how='right')

# Perform the aggregation to road network level

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

# re classify the road class
print(cycle['roadclassi'].unique())
# ['Unknown' 'Not Classified' 'Unclassified' 'B Road' 'A Road' 'Classified Unnumbered' 'Motorway']
print(cycle['directiona'].unique())
# ['bothDirections' 'inOppositeDirection' 'inDirection']

cycle['classification'] = cycle['roadclassi'].replace(
    {'Unknown': 'Other', 'Not Classified': 'Other', 'Unclassified': 'Other', 'Classified Unnumbered': 'Other'})
cycle.drop(columns=['roadclassi', 'index', 'routehiera'], inplace=True)

diff = 'diff_03/01/22'
cycle10 = cycle[
    ['toid', 'geometry', 'directiona', 'length', 'roadwidthm', 'elevationg', 'classification', diff]].groupby(
    'classification').apply(lambda x: top_10_abs(x, diff)).reset_index(drop=True)

# plot the top 25% of the absolute values (flow changes) for each road class
cycle10 = gpd.GeoDataFrame(cycle10, crs={'init': 'epsg:27700'}, geometry='geometry')

# label the inner and outer london to the cycle10
cycle10 = gpd.sjoin(cycle10, inoutter, how='inner', op='within')
cycle10 = cycle10.drop(columns=['index_right', 'Source', 'Area_Ha', 'Shape_Leng', 'Shape_Area'])

# visualize the top 10%
fig, ax = plt.subplots(figsize=(20, 20), dpi=500)
cycle10.plot(column='diff_03/01/22', legend=True, cmap='RdBu_r', ax=ax, linewidth=0.1)
ax.set_axis_off()
plt.show()
