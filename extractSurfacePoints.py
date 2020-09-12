import rasterio
from rasterio.warp import transform
import os
import numpy as np
import math
import csv
import cv2
import matplotlib.pyplot as plt
import re

# read GeoTiff Data using rasterio
dataset = rasterio.open('TroutPassAerial4326.tiff')
print(dataset.profile)

r = dataset.read(1)
g = dataset.read(2)
b = dataset.read(3)
# print(r)
# print(g)
# print(b)

print(dataset.xy(0, 0))
print(dataset.xy(dataset.height, dataset.width))

bounds = dataset.bounds
xr = np.linspace(bounds.left, bounds.right, dataset.width)
yr = np.linspace(bounds.top, bounds.bottom, dataset.height)
x, y = np.meshgrid(xr, yr)

lon, lat = transform(dataset.crs, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())

lon = np.asarray(lon).reshape((dataset.height, dataset.width))
lat = np.asarray(lat).reshape((dataset.height, dataset.width))

# print(lon)
# print(lat)

# read hgt data
fn = 'N38W079.hgt'
matched = re.match(r'N(.*)W(.*).hgt', fn, re.M|re.I)
if matched:
    start_lon = -int(matched.group(2))
    start_lat = int(matched.group(1))
else:
    start_lon = -79
    start_lat = 38

siz = os.path.getsize(fn)
dim = int(math.sqrt(siz/2))

assert dim*dim*2 == siz, 'Invalid file size'

hgt_data = np.fromfile(fn, np.dtype('>i2'), dim*dim).reshape((dim, dim))

hgt_data = hgt_data.astype('float')

sigma_y = 40.0
sigma_x = 40.0
sigma = [sigma_y, sigma_x]

# hgt_data = sp.ndimage.filters.gaussian_filter(hgt_data, sigma, mode='constant')

hgt_data = cv2.GaussianBlur(hgt_data, (5, 5), 0)

# clip data
bound_x, bound_y = transform(dataset.crs, {'init': 'EPSG:4326'},
                     [bounds.left, bounds.right], [bounds.top, bounds.bottom])

startX = int((bound_x[0] - start_lon) * 3600)
endX = int((bound_x[1] - start_lon) * 3600)

startY = int((bound_y[0] - start_lat) * 3600)
endY = int((bound_y[1] - start_lat) * 3600)

clipped = hgt_data[endY:startY, startX:endX]

# upsampling
resized = cv2.resize(clipped, (dataset.width, dataset.height), interpolation=cv2.INTER_CUBIC)
resized = np.flipud(resized)

# Write to CSV file
file = open('geo_spatial.csv', 'w', newline='')
writer = csv.writer(file)
writer.writerow(["X", "Y", "Z", "R", "G", "B"])

idx = 0
idy = 0
for r1, g1, b1, x1, y1, lon1, lat1 in zip(r, g, b, x, y, lon, lat):
    idx = 0
    for r2, g2, b2, x2, y2, lon2, lat2 in zip(r1, g1, b1, x1, y1, lon1, lat1):
        z2 = resized[idy][idx]
        idx += 1

        # print(x2, y2, z2, r2, g2, b2)
        writer.writerow([x2, y2, z2, r2, g2, b2])

    idy += 1

file.close()

print(idx, idy)

# Plot X,Y,Z
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, resized, 50, cmap='binary')

# ax.plot_surface(x, y, resized, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# input
input1 = input()
