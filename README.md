# slopefinder

### Provide a lat lon and find out which way the ground surface slopes

## Notes on Usage:  
- Can be used one of three ways:  
    - Call from the command line. For example: `python slopefinder.py -lat 40.363124 -lon -74.173452`  
    - Use the python function directly. For example: `slope_direction = slopefinder.find_slope_direction(40.363124, -74.173452)` (with the correct imports)  
    - Run it as a Flask app. `python server.py` then "http://127.0.0.1:5000/find_slope_direction?lat=40.363124&lon=-74.173452"  
- What's returned? The direction the ground surface slopes (aspect) in degrees clockwise from North (i.e. if your point was on a hill with uphill being west and downhill east, it will return `90`)  
- The dependencies (specifically GDAL) can be a bit finicky to install. The environment.yml file and conda get you most of the way there.

### Acknowledgments:
- https://github.com/bopen/elevation
- https://gist.github.com/jkatagi/a1207eee32463efd06fb57676dcf86c8
