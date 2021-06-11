import flask
from flask import request
import slopefinder

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/find_slope_direction', methods=['GET'])
def home():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    slope_direction = slopefinder.find_slope_direction(lat, lon)

    return {"slope_direction": slope_direction}

app.run()
