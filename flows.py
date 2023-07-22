import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import gzip
import shutil
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


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

# return the top 10% of absolute values
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
tube_line = 'https://raw.githubusercontent.com/oobrien/vis/master/tubecreature/data/tfl_lines.json'
tube_station = 'https://raw.githubusercontent.com/oobrien/vis/master/tubecreature/data/tfl_stations.json'

inoutter = gpd.read_file(inoutter)
inoutter.to_crs(epsg=27700, inplace=True)

tube_station = gpd.read_file(tube_station)
tube_station.to_crs(epsg=27700, inplace=True)
tube_station = gpd.sjoin(tube_station, inoutter, op='within')

tube_line = gpd.read_file(tube_line)
tube_line.to_crs(epsg=27700, inplace=True)
tube_line = gpd.sjoin(tube_line, inoutter, op='within')

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
# cycle10 = cycle[
#     ['toid', 'geometry', 'directiona', 'length', 'roadwidthm', 'elevationg', 'classification', diff]].groupby(
#     'classification').apply(lambda x: top_10_abs(x, diff)).reset_index(drop=True)

cycle10 = cycle[cycle[diff] >= cycle[diff].quantile(0.9)].reset_index(drop=True)

# plot the top 10% of the absolute values (flow changes) for each road class
cycle10 = gpd.GeoDataFrame(cycle10, crs={'init': 'epsg:27700'}, geometry='geometry')

# label the inner and outer london to the cycle10
cycle10 = gpd.sjoin(cycle10, inoutter, how='inner', op='within')
cycle10 = cycle10.drop(columns=['index_right', 'Source', 'Area_Ha', 'Shape_Leng', 'Shape_Area'])

# visualize the top 10%
# fig, ax = plt.subplots()
# cycle10.plot(column='diff_03/01/22', legend=True, cmap='viridis', ax=ax)
# ax.set_axis_off()
# plt.show()
# plt.savefig('cycle_top10.png', dpi=600)


fig, ax = plt.subplots(dpi=600)
# Get unique types
# road_class = cycle10['classification'].unique()
# road_class = ['Motorway', 'A Road', 'B Road', 'Other']
road_class = ['Other', 'B Road', 'A Road', 'Motorway']
widths = {
    'Other': 0.5,
    'B Road': 1,
    'A Road': 1.5,
    'Motorway': 2
}

# Generate a colormap with the number of colors equal to the number of types
cmap = plt.cm.get_cmap('tab10_r', len(road_class))

inoutter.boundary.plot(color='black', ax=ax, linewidth=1)
tube_line['geometry'] = tube_line.geometry.buffer(500)

tube_line.plot(color='gainsboro', ax=ax, legend=True)

for i, class_ in enumerate(road_class):
    # Subset data by type
    data = cycle10[cycle10['classification'] == class_]
    # Plot subset with a unique color
    data.plot(column='date', color=cmap(i), ax=ax, label=class_, linewidth=widths[class_])

ax.set_axis_off()
plt.legend(loc='lower right')
plt.show()


fig, ax = plt.subplots(dpi=600)
# 创建一个新的列用于存储线宽
linewidth_map = {'Motorway': 2, 'A Road': 1.5, 'B Road': 1, 'Other': 0.5}
cycle10['linewidth'] = cycle10['classification'].map(linewidth_map)

# 根据'linewidth'列进行排序
cycle10 = cycle10.sort_values('linewidth')

# 绘制图形
cycle10.plot(column=diff, ax=ax, cmap='Wistia', linewidth=cycle10['linewidth'], legend=True)

custom_lines = [Line2D([0], [0], color='b', lw=width) for width in linewidth_map.values()]
ax.legend(custom_lines, linewidth_map.keys())
ax.set_axis_off()
plt.show()

