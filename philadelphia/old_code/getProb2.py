import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
def interpolate_probabilities(gdf, points='auto'):
    gdf = gdf.to_crs(epsg=3857)

    if points == 'auto':
        x = np.linspace(gdf.total_bounds[0], gdf.total_bounds[2], 500)
        y = np.linspace(gdf.total_bounds[1], gdf.total_bounds[3], 500)
    else:
        x = np.linspace(gdf.total_bounds[0], gdf.total_bounds[2], points)
        y = np.linspace(gdf.total_bounds[1], gdf.total_bounds[3], points)

    xx, yy = np.meshgrid(x, y)
    points = np.vstack((gdf.geometry.x, gdf.geometry.y)).T
    values = gdf['normalized_probability']

    # Use linear interpolation to avoid NaNs
    interpolated_probabilities = griddata(points, values, (xx, yy), method='linear')

    # Fill NaN values with zero or another appropriate value
    interpolated_probabilities = np.nan_to_num(interpolated_probabilities, nan=0.0)

    return xx, yy, interpolated_probabilities



def calculate_and_normalize_probabilities(data, covariance_matrix):
    # Calculate the Gaussian probabilities for each point
    coordinates = np.vstack((data['lat'], data['lon'])).T
    probabilities = np.zeros(len(data))

    for coordinate in coordinates:
        rv = multivariate_normal(coordinate, covariance_matrix)
        probabilities += rv.pdf(coordinates)

    # Normalize the probabilities within the group
    return probabilities / probabilities.sum()

def main(file_path, covariance_value):
    data = pd.read_csv(file_path)

    # Define the covariance matrix
    covariance_matrix = np.array([[covariance_value, 0], [0, covariance_value]])

    # Initialize a column for normalized probabilities
    data['normalized_probability'] = 0

    # Calculate and normalize probabilities separately for each event type
    for typ, group_data in data.groupby('typ'):
        probabilities = calculate_and_normalize_probabilities(group_data, covariance_matrix)
        data.loc[group_data.index, 'normalized_probability'] = probabilities

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(data.lon, data.lat)
    )
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=3857)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    for typ, group_data in gdf.groupby('typ'):
        xx, yy, interpolated_probabilities = interpolate_probabilities(group_data)
        print(interpolated_probabilities)
        #levels = np.linspace(0, 1, 10)
        ax.contour(xx, yy, interpolated_probabilities, levels=10, cmap='viridis', alpha=0.5)
    
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
    ax.set_axis_off()
    plt.title('Smoothed Probability Distribution of Events by Type with Basemap')
    plt.savefig(file_path.replace('.csv','.png'),dpi=300,bbox_inches='tight',transparent=True)
    return data

if __name__ == "__main__":
    file_path = 'predictions/prediction_2024-04-16.csv'
    covariance_value = 0.0001  # Adjust as needed
    data=main(file_path, covariance_value)
    data.to_csv(file_path.replace('.csv','mod.csv'))
