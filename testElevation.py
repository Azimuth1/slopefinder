#testElevation.py

import elevation
import os, sys

# clip the SRTM1 30m DEM of Rome and save it to Rome-DEM.tif
#elevation.clip(bounds=(12.35, 41.8, 12.65, 42), output='Rome-DEM.tif')
elevation.clip(bounds=(-78.92, 38.95, -78.87, 39.00), output=os.path.join(sys.path[0], "Python-TroutPass-DEM.tif"))
# clean up stale temporary files and fix the cache in the event of a server error
#elevation.clean()
