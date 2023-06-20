import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
import matplotlib.dates as mdates
import datetime

# load the data

# from flows import flows
# from LSOA_related import lsoa

flows_lsoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/processed data/flows_lsoa.shp'
flows_lsoa = gpd.read_file(flows_lsoa, crs={'init': 'epsg:27700'})
inoutter = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/lp-consultation-oct-2009-inner-outer-london-shp/lp-consultation-oct-2009-inner-outer-london.shp'
inoutter = gpd.read_file(inoutter, crs={'init': 'epsg:27700'})
flows_lsoa = gpd.sjoin(flows_lsoa, inoutter, how='inner', op='within')
flows_lsoa.drop(columns=['index_right', 'Source', 'Area_Ha', 'Shape_Leng', 'Shape_Area'], inplace=True)
flows_lsoa.rename(columns={'Boundary': 'Inner_Outer'}, inplace=True)

print(flows_lsoa.head())
print(flows_lsoa.columns)
print(len(flows_lsoa['LSOA21CD'].unique()))
print(len(flows_lsoa['date'].unique()))
print(flows_lsoa.describe())

# split the data into train and test
train = flows_lsoa.iloc[:int(0.8 * (len(flows_lsoa)))]  # 80% of the data as train
test = flows_lsoa.iloc[int(0.8 * (len(flows_lsoa))):]  # 20% of the data as test

flows_londonsum = flows_lsoa.groupby('date').aggregate(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'})
flows_londonsum.index = pd.to_datetime(flows_londonsum.index)

flows_inouttersum = flows_lsoa.groupby(['date', 'Inner_Outer']).aggregate(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'}).reset_index()
flows_inner = flows_inouttersum[flows_inouttersum['Inner_Outer'] == 'Inner London'].drop(columns=['Inner_Outer'])
flows_outer = flows_inouttersum[flows_inouttersum['Inner_Outer'] == 'Outer London'].drop(columns=['Inner_Outer'])

flows_inner.set_index('date', inplace=True)
flows_outer.set_index('date', inplace=True)
flows_inner.index = pd.to_datetime(flows_londonsum.index)
flows_outer.index = pd.to_datetime(flows_londonsum.index)

# visualize the original temporal data
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# Add a shaded region
start_date = datetime.datetime(2022, 2, 28)
end_date = datetime.datetime(2022, 3, 6)
strike_start = datetime.datetime(2022, 3, 1)
strike_end = datetime.datetime(2022, 3, 2)
ax1.axvspan(start_date, end_date, color='gray', alpha=0.1, edgecolor='none')
ax1.axvspan(strike_start, strike_end, color='maroon', alpha=0.1, edgecolor='none')
# Add text
text_x = start_date + (end_date - start_date) / 2  # Center of the shaded region
text_y = float(flows_londonsum.max().max()) / 2  # Middle of the y range
text_y = np.power(10, text_y)  # Convert the y-value from logarithmic to original scale
ax1.text(text_x, text_y, 'Strike week', horizontalalignment='center', verticalalignment='center', color='maroon',
         fontsize=12)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

cmap_o = cm.get_cmap('tab20', len(flows_inner.columns))

for i in range(len(flows_inner.columns)):
    ax1.plot(flows_inner.index, flows_inner.iloc[:, i], color=cmap_o(i), label=flows_inner.columns[i],
             linewidth=1.5, linestyle='-', marker='o')
for i in range(len(flows_outer.columns)):
    ax2.plot(flows_outer.index, flows_outer.iloc[:, i], color=cmap_o(i), label=flows_outer.columns[i],
             linewidth=1.5, linestyle='--', marker='o')

ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_xlabel('Date')
ax1.set_ylabel('Inner London flows change for different modes')
ax2.set_ylabel('Outter London flows change for different modes')
ax1.legend(loc='upper left', title='Inner London')
ax2.legend(loc='upper right', title='Outter London')
plt.title('Inner & Outter London flows change for different modes')
# rotate the x-axis ticks
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig('/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Outout/pics/Inner & Outter London flows change for different modes', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
# Add a shaded region
start_date = datetime.datetime(2022, 2, 28)
end_date = datetime.datetime(2022, 3, 6)
strike_start = datetime.datetime(2022, 3, 1)
strike_end = datetime.datetime(2022, 3, 2)
ax.axvspan(start_date, end_date, color='gray', alpha=0.1, edgecolor='none')
ax.axvspan(strike_start, strike_end, color='maroon', alpha=0.1, edgecolor='none')
# Add text
text_x = start_date + (end_date - start_date) / 2  # Center of the shaded region
text_y = float(flows_londonsum.max().max()) / 2  # Middle of the y range
text_y = np.power(10, text_y)  # Convert the y-value from logarithmic to original scale
ax1.text(text_x, text_y, 'Strike week', horizontalalignment='center', verticalalignment='center', color='maroon',
         fontsize=12)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

for i in range(len(flows_londonsum.columns)):
    ax.plot(flows_londonsum.index, flows_londonsum.iloc[:, i], color=cmap_o(i), label=flows_londonsum.columns[i],
            linewidth=1.5, marker='o')
ax.set_yscale('log')
ax.set_xlabel('Date')
ax.legend(loc='upper left', title='Inner London')

# rotate the x-axis ticks
fig.autofmt_xdate()
fig.tight_layout()
plt.title('Total flows change for different modes in London')
fig.savefig('/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Outout/pics/Total flows change for different modes in London', dpi=300, bbox_inches='tight')
plt.show()