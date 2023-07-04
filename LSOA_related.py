import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import gzip
import shutil


def unzip_files_in_folder(folder_path):
    # Get the list of files in the folder
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        if file_name.endswith('.gz'):
            file_path = os.path.join(folder_path, file_name)

            # Create a new file name by removing the .gz extension
            output_file_path = file_path[:-3]

            # Open the input gzipped file and the output file
            with gzip.open(file_path, 'rb') as gz_file, open(output_file_path, 'wb') as out_file:
                shutil.copyfileobj(gz_file, out_file)

            print(f"Unzipped: {file_name} -> {output_file_path}")


# # Unzip the traffic flow data
# folder_path = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]Traffic flow'
# unzip_files_in_folder(folder_path)

# Process the traffic flow data and network data
# # read the road network data
# road_network = gpd.read_file('/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]roadnet_london/roadnet_london.shp')
# road_network.crs = {'init': 'epsg:27700'}
# road_network.plot()
# plt.show()


# Procession for the LSOA-related data
# # read the data
london = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/london_boundary/greater_london_const_region.shp'
lsoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/london_LSOA/london_LSOA.shp'
msoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/MSOA/MSOA.shp'
oacode = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/OA codes.csv'
population_num = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS001 - Number of usual residents in households and communal establishments.csv'
population = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS006 - Population density.csv'
num_of_households = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS041 - Number of Households.csv'
household_with_child = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/RM132 - Tenure by dependent children in household.csv'
age = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS007A - Age by five-year age bands .csv'
ethnic = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS021 - Ethnic group .csv'
gender = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS008 - Sex.csv'

jobs = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/jobs density&numbers.csv'
IMD = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/IMD_LSOA/IMD_LSOA.shp'
household_income = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/incomeMSOA2018.xls'
household_car = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS045 - Car or van availability.csv'
economic_activity = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/TS066 - Economic activity status.csv'

land_cover = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/land cover/land cover.shp'
road_network = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_network.shp'
intersection = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_network_intersection/intersection_count_lsoa/intersection_count_lsoa.shp'

POI = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/POI_london/POI_spatialJoin_LSOA.shp'
naptan = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/NaPTAN/NaPTAN_spatialJoin.csv'
road_length = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/[XH]road_network/road_length_lsoa.csv'

# # Load data
# london = gpd.read_file(london, crs={'init': 'epsg:27700'})
lsoa = gpd.read_file(lsoa, crs={'init': 'epsg:27700'})
# msoa = gpd.read_file(msoa, crs={'init': 'epsg:27700'})
oacode = pd.read_csv(oacode, encoding='latin1', low_memory=False)

population_num = pd.read_csv(population_num, skiprows=lambda x: x < 6)
population = pd.read_csv(population, skiprows=lambda x: x < 6)
num_of_households = pd.read_csv(num_of_households, skiprows=lambda x: x < 6)
household_with_child = pd.read_csv(household_with_child, skiprows=lambda x: x < 7)
age = pd.read_csv(age, skiprows=lambda x: x < 4)
ethnic = pd.read_csv(ethnic, skiprows=lambda x: x < 6, low_memory=False)
gender = pd.read_csv(gender, skiprows=lambda x: x < 6)

jobs = pd.read_csv(jobs, skiprows=lambda x: x < 5)
IMD = gpd.read_file(IMD, crs={'init': 'epsg:27700'})
household_income = pd.read_excel(household_income, sheet_name='Total annual income', skiprows=lambda x: x < 4)
household_car = pd.read_csv(household_car, skiprows=lambda x: x < 6)
economic_activity = pd.read_csv(economic_activity, skiprows=lambda x: x < 6)

# land_cover = gpd.read_file(land_cover, crs={'init': 'epsg:27700'})
road_network = gpd.read_file(road_network, crs={'init': 'epsg:27700'})
intersection = gpd.read_file(intersection, crs={'init': 'epsg:27700'})
POI = gpd.read_file(POI, crs={'init': 'epsg:27700'})
naptan = pd.read_csv(naptan, low_memory=False)
road_length = pd.read_csv(road_length, low_memory=False).astype({'length': 'float64'})
'''london.plot()
lsoa.plot()
msoa.plot()
IMD.plot()
land_cover.plot()
road_network.plot()
POI.plot()
plt.show()'''

# # build the indicators and link them to the LSOA

lsoa = pd.merge(lsoa,
                oacode[['lsoa21cd', 'msoa21cd', 'msoa21nm', 'lad22cd', 'lad22nm']].drop_duplicates(subset='lsoa21cd'),
                left_on='LSOA21CD',
                right_on='lsoa21cd', how='left')

lsoa.rename(columns={'msoa21cd': 'MSOA21CD', 'msoa21nm': 'MSOA21NM', 'lad22cd': 'LAD22CD', 'lad22nm': 'LAD22NM'},
            inplace=True)

# # # Demographic indicators
# lsoa['area(km2)'] = lsoa['geometry'].area / 1000000
lsoa = pd.merge(lsoa, population_num[['mnemonic', '2021']], left_on='LSOA21CD', right_on='mnemonic', how='left')
lsoa = lsoa.rename(columns={'2021': 'Population'})
population['Population density (p/ha)'] = population['Usual residents per square kilometre'] / 100
lsoa = pd.merge(lsoa, population[['mnemonic', 'Population density (p/ha)']], left_on='LSOA21CD',
                right_on='mnemonic', how='left')

lsoa = pd.merge(lsoa, num_of_households[['mnemonic', 'Number of households']], left_on='LSOA21CD', right_on='mnemonic',
                how='left')
lsoa = pd.merge(lsoa, household_with_child[['mnemonic', '2021']], left_on='LSOA21CD', right_on='mnemonic', how='left')

age['Pct of under 19 (%)'] = age['%.1'] + age['%.2'] + age['%.3'] + age['%.4']
age['Pct of 20-34 (%)'] = age['%.5'] + age['%.6'] + age['%.7']
age['Pct of 35-49 (%)'] = age['%.8'] + age['%.9'] + age['%.10']
age['Pct of 50-64 (%)'] = age['%.11'] + age['%.12'] + age['%.13']
age['Pct of over 65 (%)'] = age['%.14'] + age['%.15'] + age['%.16'] + age['%.17'] + age['%.18']
age['totalPct'] = age['Pct of under 19 (%)'] + age['Pct of 20-34 (%)'] + age['Pct of 35-49 (%)'] + age[
    'Pct of 50-64 (%)'] + age['Pct of over 65 (%)']
lsoa = pd.merge(lsoa, age[
    ['mnemonic', 'Pct of under 19 (%)', 'Pct of 20-34 (%)', 'Pct of 35-49 (%)', 'Pct of 50-64 (%)',
     'Pct of over 65 (%)']], left_on='LSOA21CD', right_on='mnemonic', how='left')

lsoa = pd.merge(lsoa, gender[['mnemonic', '%.1', '%.2']], left_on='LSOA21CD', right_on='mnemonic', how='left')
ethnic['Pct of Caucasian (%)'], ethnic['Other ethnic groups (%)'] = ethnic['%.16'], ethnic['%.1'] + ethnic['%.7'] + \
                                                                                    ethnic['%.11'] + ethnic['%.22']
lsoa = pd.merge(lsoa, ethnic[['Pct of Caucasian (%)', 'Other ethnic groups (%)', 'mnemonic']], left_on='LSOA21CD',
                right_on='mnemonic', how='left')

lsoa = lsoa.rename(
    columns={'2021': 'num_household_with_child', '%.1': 'Pct of female (%)', '%.2': 'Pct of male (%)'})

# # # Socioeconomic indicators
lsoa = pd.merge(lsoa, jobs[['Total jobs', 'Jobs density', 'mnemonic']], left_on='LAD22CD', right_on='mnemonic',
                how='left')  # contains nan values
lsoa = pd.merge(lsoa, IMD[['IMDRank', 'IMDDecil', 'LSOA21CD']], left_on='LSOA21CD', right_on='LSOA21CD', how='left')
lsoa = pd.merge(lsoa, household_income[['MSOA code', 'Total annual income (£)']], left_on='MSOA21CD',
                right_on='MSOA code', how='left')  # contains nan values due to the missing MSOA code
lsoa = pd.merge(lsoa, household_car[['mnemonic', '%.1', '%.2', '%.3', '%.4']], left_on='LSOA21CD', right_on='mnemonic',
                how='left')

lsoa = lsoa.rename(columns={'%.1': 'No cars or vans in household (%)', '%.2': '1 car or van in household (%)',
                            '%.3': '2 cars or vans in household (%)', '%.4': '3 or more cars or vans in household (%)',
                            'Total annual income (£)': 'Average household income (£)'})
lsoa = pd.merge(lsoa, economic_activity[['mnemonic', '%.1', '%.2']], left_on='LSOA21CD', right_on='mnemonic',
                how='left')
lsoa = lsoa.rename(columns={'%.1': 'Economically active (excluding full-time students) (%)',
                            '%.2': 'Economically active and a full-time student (%)'})

lsoa['Pct of households with child (%)'] = lsoa['num_household_with_child'] / lsoa['Number of households'] * 100

# # #  Built environment indicators
# intersections_in_lsoa = gpd.sjoin(intersection,lsoa, how='inner', op='within')
# points_per_polygon = points_in_polygons.groupby('index_right').size()

# number of intersections
lsoa = pd.merge(lsoa, intersection[['LSOA21CD', 'Join_Cou_1']], left_on='LSOA21CD', right_on='LSOA21CD', how='left')
lsoa = lsoa.rename(columns={'Join_Cou_1': 'Number of intersections'})

# Pct of POIs
counts = POI.groupby(['LSOA21CD', 'groupname']).size().reset_index(name='POI_count')
counts_wide = counts.pivot(index='LSOA21CD', columns='groupname', values='POI_count').reset_index()
counts_wide.fillna(0, inplace=True)

row_sums = counts_wide.iloc[:, 1:].sum(axis=1)
proportions = counts_wide.iloc[:, 1:].apply(lambda x: x / row_sums * 100)

proportions['LSOA21CD'] = counts_wide['LSOA21CD']
cols = proportions.columns.tolist()
cols = cols[-1:] + cols[:-1]
proportions = proportions[cols]
proportions = proportions.rename(columns=lambda x: 'Pct of ' + x + ' POIs (%)' if x != 'LSOA21CD' else x)

lsoa = pd.merge(lsoa, proportions, left_on='LSOA21CD', right_on='LSOA21CD', how='left')
lsoa.fillna(0, inplace=True)

lsoa.drop(columns=['lsoa21cd', 'mnemonic_x', 'mnemonic_y', 'num_household_with_child'], inplace=True)

# NaPTAN stops count
counts_naptan = naptan.groupby(['LSOA21CD']).size().reset_index(name='NaPTAN_count')
counts_naptan.fillna(0, inplace=True)
lsoa = pd.merge(lsoa, counts_naptan[['LSOA21CD', 'NaPTAN_count']], left_on='LSOA21CD', right_on='LSOA21CD', how='left')
lsoa['Public transit access point density (n/ha)'] = lsoa['NaPTAN_count'] / lsoa.area * 10000

# road density
sum_length = road_length.groupby(['LSOA21CD']).sum().reset_index()
sum_length['Street density (m/ha)'] = sum_length['length'] / lsoa.area.astype(float) * 10000
lsoa = pd.merge(lsoa, sum_length[['LSOA21CD', 'Street density (m/ha)']], left_on='LSOA21CD', right_on='LSOA21CD',
                how='left')

# connect the flows to the LSOA

lsoa.to_file('/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/processed data/Mosaic_LSOA.shp')

lsoa.plot(column='Pct of Retail POIs (%)', legend=True)
plt.axis('off')
plt.show()
