import rasterio
from rasterio.warp import transform
import os, sys
import numpy as np
import math
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import elevation

# Count the arguments
arguments = len(sys.argv) - 1
if arguments <> 2
    print "Usage: extractGwFlow.py file=<filepath> output=<path_to_output>"

output_file = os.path.join(sys.path[0], "Python-TroutPass-DEM.tif")
# overwrite output_file with command line Parameter
# read the CSV file
# get the lat long bounds of the points
temp_bounds=(-78.92, 38.95, -78.87, 39.00)
# expand the bounds by a bit
# download the 3DEP SRTM1 for the area, save as a temp GeoTiff


# Output argument-wise
position = 1
while (arguments >= position):
    print ("Parameter %i: %s" % (position, sys.argv[position]))
    position = position + 1


elevation.clip(bounds=temp_bounds, output=output_file)
# clean up stale temporary files and fix the cache in the event of a server error
elevation.clean()


# read GeoTiff Data using rasterio
aerial = rasterio.open('RandomAerial4326.tiff')
print(aerial.profile)

r = aerial.read(1)
g = aerial.read(2)
b = aerial.read(3)

print(aerial.xy(0, 0))
print(aerial.xy(aerial.height, aerial.width))

bounds = aerial.bounds
xr = np.linspace(bounds.left, bounds.right, aerial.width)
yr = np.linspace(bounds.top, bounds.bottom, aerial.height)
x, y = np.meshgrid(xr, yr)

lon, lat = transform(aerial.crs, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())

lon = np.asarray(lon).reshape((aerial.height, aerial.width))
lat = np.asarray(lat).reshape((aerial.height, aerial.width))

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
bound_x, bound_y = transform(aerial.crs, {'init': 'EPSG:4326'},
                     [bounds.left, bounds.right], [bounds.top, bounds.bottom])

startX = int((bound_x[0] - start_lon) * 3600)
endX = int((bound_x[1] - start_lon) * 3600)

startY = max(3601 - int((bound_y[0] - start_lat) * 3600), 0)
endY = 3601 - int((bound_y[1] - start_lat) * 3600)

clipped = hgt_data[startY:endY, startX:endX]

# upsampling
resized = cv2.resize(clipped, (aerial.width, aerial.height), interpolation=cv2.INTER_CUBIC)
# resized = np.flipud(resized)

# Write to CSV file
file = open('heightfield.csv', 'w', newline='')
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
        writer.writerow([lon2, lat2, z2, r2, g2, b2])

    idy += 1

file.close()

# print(idx, idy)

# Plot X,Y,Z
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].imshow(hgt_data)

rect = patches.Rectangle((startX,startY), endX - startX, endY - startY,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax[0, 0].add_patch(rect)
ax[0, 1].imshow(clipped)
ax[1, 0].imshow(resized)

rgb = np.dstack((r,g,b))
ax[1, 1].imshow(rgb)

plt.show()
