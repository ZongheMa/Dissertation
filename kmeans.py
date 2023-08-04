import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
import IPython.display as display
import copy
import seaborn as sns


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

traffic_flows = merge_csv_files(
    '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]Traffic flow')
road_network = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_network.shp'

# clean the traffic flow data
traffic_flows = traffic_flows.drop_duplicates(['toid', 'date'])
traffic_flows = traffic_flows.groupby(['toid', 'date']).agg(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'}).reset_index()

traffic_flows['total'] = traffic_flows['bus'] + traffic_flows['car'] + traffic_flows['cycle'] + traffic_flows[
    'walks'] + traffic_flows['stationary']

lsoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/london_LSOA/london_LSOA.shp'
road_network = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_network.shp'
inoutter = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/lp-consultation-oct-2009-inner-outer-london-shp/lp-consultation-oct-2009-inner-outer-london.shp'
tube_line = 'https://raw.githubusercontent.com/oobrien/vis/master/tubecreature/data/tfl_lines.json'
tube_station = 'https://raw.githubusercontent.com/oobrien/vis/master/tubecreature/data/tfl_stations.json'

inoutter = gpd.read_file(inoutter)
inoutter.to_crs(epsg=27700, inplace=True)

# tube_station = gpd.read_file(tube_station)
# tube_station.to_crs(epsg=27700, inplace=True)
# tube_station = gpd.sjoin(tube_station, inoutter, op='within')

tube_line = gpd.read_file(tube_line)
tube_line.to_crs(epsg=27700, inplace=True)
tube_line = gpd.sjoin(tube_line, inoutter, op='within')

lsoa = gpd.read_file(lsoa, crs={'init': 'epsg:27700'})
road_network = gpd.read_file(road_network, crs={'init': 'epsg:27700'})

flows = pd.merge(
    road_network[['toid', 'roadclassi', 'geometry', 'directiona', 'length', 'roadwidthm', 'elevationg'
                  ]],
    traffic_flows, left_on='toid', right_on='toid', how='right')

flows['classification'] = flows['roadclassi'].replace(
    {'Unknown': 'Local Road', 'Not Classified': 'Local Road', 'Unclassified': 'Local Road',
     'Classified Unnumbered': 'Local Road'})

flows.drop(columns=['roadclassi'], inplace=True)

stage_date = ['03/01/22', '02/22/22', '03/08/22']
flows = flows.loc[flows['date'].isin(stage_date)]

# label the regional level
flows = gpd.sjoin(flows, inoutter, how='inner', predicate='within')
flows = flows.drop(columns=['index_right', 'Source', 'Area_Ha', 'Shape_Leng', 'Shape_Area'])

# convert the dataframe
flows = pd.melt(flows,
                id_vars=['toid', 'classification', 'geometry', 'directiona', 'length', 'roadwidthm', 'elevationg',
                         'date', 'Boundary'], var_name='mode', value_name='flow')

flows = pd.pivot_table(flows,
                       index=['toid', 'classification', 'geometry', 'directiona', 'length', 'roadwidthm', 'elevationg',
                              'date', 'Boundary', 'mode'], columns='date', values='flow', aggfunc='first').reset_index()

flows.drop(columns=['date'], inplace=True)


flows = flows.groupby(
    ['toid', 'mode', 'classification', 'geometry', 'directiona', 'length', 'roadwidthm', 'elevationg',
     'Boundary'], as_index=False).agg(
    {'03/01/22': 'first', '02/22/22': 'first', '03/08/22': 'first'}).reset_index()
# calculate the impact and recovery flows for one strike
flows['impact_flow'] = flows['03/01/22'] - flows['02/22/22']
flows['recovery_flow'] = flows['03/08/22'] - flows['03/01/22']

flows.drop(columns=['index'],inplace=True)

# calculate the impact and recovery flows for one strike
flows['impact_flow'] = flows['03/01/22'] - flows['02/22/22']
flows['recovery_flow'] = flows['03/08/22'] - flows['03/01/22']



from sklearn.cluster import KMeans

# Group the DataFrame by unique values in 'mode' column
grouped = flows.groupby('mode')

# Perform KMeans clustering for each group separately
for mode, group in grouped:
    kmeans = KMeans(n_clusters=7, random_state=42)
    flows.loc[group.index, 'KmeansBymode_impact'] = kmeans.fit_predict(group[['impact_flow']])
    flows.loc[group.index, 'KmeansBymode_recovery'] = kmeans.fit_predict(group[['recovery_flow']])

flows_shp = gpd.GeoDataFrame(flows, geometry='geometry', crs={'init': 'epsg:27700'})
flows_shp.to_file('/Users/zonghe/Desktop/flows.shp')

# Create a scatter plot to visualize 'KmeansBymode_impact' and its clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=flows, x='KmeansBymode_impact', y='impact_flow', hue='mode', palette='tab10', s=100)
plt.title('KMeans Clustering for Different Modes')
plt.xlabel('KmeansBymode_impact')
plt.ylabel('Impact Flow')
plt.legend(title='Mode', loc='upper right', bbox_to_anchor=(1.15, 1.0))
plt.show()



def visualize_kmeans_by_mode_and_classification(flows, feature1, feature2):
    """
    Visualize KMeans clustering results for different 'mode' and 'classification' categories.

    Parameters:
        flows (DataFrame): The DataFrame containing the data.
        feature1 (str): The name of the first feature to visualize KMeans clusters.
        feature2 (str): The name of the second feature to visualize KMeans clusters.

    Returns:
        None
    """

    # Create a new column to store the combined KMeans cluster labels
    flows['KmeansByModeAndClassification'] = None

    # Group the DataFrame by unique values in 'mode' column
    grouped_by_mode = flows.groupby('mode')

    # Perform KMeans clustering for each 'mode' group separately
    for mode, group_mode in grouped_by_mode:
        kmeans = KMeans(n_clusters=6, random_state=42)
        flows.loc[group_mode.index, 'KmeansByModeAndClassification'] = kmeans.fit_predict(group_mode[[feature1, feature2]])

        # Create subplots for each 'classification' category
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=flows.loc[group_mode.index], x=feature1, y=feature2, hue='KmeansByModeAndClassification', palette='tab10', s=100)
        plt.title(f'KMeans Clustering for {mode} Mode')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.legend(title='KMeans Cluster')
        plt.show()

# Call the visualization method
visualize_kmeans_by_mode_and_classification(flows, 'impact_flow', 'recovery_flow')


# 定义分类顺序
classification_order = ['Motorway', 'A Road', 'B Road', 'Local Road']
mode_order = ['total', 'car', 'bus', 'cycle', 'walks', 'stationary']

# 使用pivot_table将'mode'作为列，'classification'作为行，'impact_flow'作为值，并进行求和（sum）
flows_pivot = pd.pivot_table(flows, index='classification', columns='mode', values='impact_flow', aggfunc='sum')

# 对DataFrame进行排序，以调整顺序
flows_pivot = flows_pivot.reindex(classification_order, axis=0)
flows_pivot = flows_pivot[mode_order]


# 绘制条形图
plt.figure(figsize=(10, 6))
flows_pivot.plot(kind='bar', ax=plt.gca())
plt.title('Impact Flow by Classification and Mode')
plt.xlabel('Classification')
plt.ylabel('Impact Flow')
plt.legend(title='Mode', loc='upper right', bbox_to_anchor=(1.15, 1.0))
plt.show()

