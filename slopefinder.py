import tempfile
import elevation
import rasterio
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import geotiffTools as gtt
from osgeo import gdal
import argparse


def find_slope_direction(lat, lon):
    lat, lon = float(lat), float(lon)
    temp_files = create_temp_files()
    data_bounds = get_data_bounds(lat, lon)
    elevation.clip(bounds=data_bounds, output=temp_files["dem_file"])
    get_dense_dem_file(temp_files)
    aspect = calculate_aspect(temp_files["dense_dem_file"], temp_files["aspect_file"])
    asp = rasterio.open(temp_files["aspect_file"])
    slope_direction = int([x[0] for x in asp.sample([[lon,lat]])][0])
    return slope_direction


def create_temp_files():
    tempdir = tempfile.gettempdir()
    return {
        "dem_file": tempdir+"/dem_file.tiff",
        "dense_dem_file": tempdir+"/dense_dem_file.tiff",
        "aspect_file": tempdir+"/aspect_file.tiff",
        "slope_file": tempdir+"/slope_file.tiff"
        }


def get_data_bounds(lat, lon):
    temp_bounds = (lon, lat, lon, lat)
    buffer_shift = (-0.001, -0.001, 0.001, 0.001)
    data_bounds = tuple(map(lambda i, j: i + j, temp_bounds, buffer_shift))
    return data_bounds


def get_dense_dem_file(temp_files):
    with rasterio.open(temp_files["dem_file"], 'r') as dataset:
        kwds = dataset.meta.copy()
        # Change the format driver for the destination dataset to
        # 'GTiff', short for GeoTIFF.
        kwds['driver'] = 'GTiff'
        # Add GeoTIFF-specific keyword arguments.
        kwds['dtype']=rasterio.uint8
        kwds['blockxsize'] = 256
        kwds['blockysize'] = 256
        upscale_factor = 2
        kwds['width'] = dataset.width * upscale_factor
        kwds['height'] = dataset.height * upscale_factor
        kwds['count'] = 1

        data = dataset.read(  #data is a numpy array
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )
        data = np.squeeze(data)

        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        kwds['transform']=transform

        gtt.array2raster(temp_files["dense_dem_file"], temp_files["dem_file"], data, "Float32")


def calculate_aspect(DEM, outfile):
    gdal.DEMProcessing(outfile, DEM, 'aspect')
    with rasterio.open(outfile) as dataset:
        aspect=dataset.read(1)
    return aspect


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lat",
        "--latitude"
    )
    parser.add_argument(
        "-lon",
        "--longitude"
    )
    args = parser.parse_args()
    slope_direction = find_slope_direction(float(args.latitude), float(args.longitude))
    print("slope_direction:", slope_direction)