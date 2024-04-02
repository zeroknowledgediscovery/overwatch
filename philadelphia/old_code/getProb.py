import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
from shapely.geometry import Point

def calculate_probabilities(data, covariance_matrix):
    coordinates = np.vstack((data['lat'], data['lon'])).T
    probabilities = np.zeros(len(data))

    for coordinate in coordinates:
        rv = multivariate_normal(coordinate, covariance_matrix)
        probabilities += rv.pdf(coordinates)

    return probabilities

def interpolate_probabilities(gdf, points='auto'):
    if points == 'auto':
        x = np.linspace(gdf.total_bounds[0], gdf.total_bounds[2], 500)
        y = np.linspace(gdf.total_bounds[1], gdf.total_bounds[3], 500)
    else:
        x = np.linspace(gdf.total_bounds[0], gdf.total_bounds[2], points)
        y = np.linspace(gdf.total_bounds[1], gdf.total_bounds[3], points)
    xx, yy = np.meshgrid(x, y)
    interpolated_probabilities = griddata(
        gdf[['x', 'y']], gdf['normalized_probability'], (xx, yy), method='cubic')
    return xx, yy, interpolated_probabilities

def main(file_path, covariance_value):
    data = pd.read_csv(file_path)

    # Define the covariance matrix
    covariance_matrix = np.array([[covariance_value, 0], [0, covariance_value]])

    # Compute Gaussian probabilities
    data['probability'] = calculate_probabilities(data, covariance_matrix)

    # Normalize the probabilities
    data['normalized_probability'] = data['probability'] / data['probability'].sum()

    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(data.lon, data.lat)
    )
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=3857)
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    # Interpolate the probabilities
    xx, yy, interpolated_probabilities = interpolate_probabilities(gdf)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contour(xx, yy, interpolated_probabilities, levels=20, cmap='Reds',alpha=.3)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
    ax.set_axis_off()
    plt.title('Smoothed Probability Distribution of Events with Basemap')
    plt.savefig(file_path.replace('.csv','.png'),dpi=300,bbox_inches='tight',transparent=True)
    return data

if __name__ == "__main__":
    file_path = 'predictions/prediction_2024-04-16.csv'
    covariance_value = 0.0001  # Change this value as needed
    data=main(file_path, covariance_value)
    data.to_csv(file_path.replace('.csv','mod.csv'))
