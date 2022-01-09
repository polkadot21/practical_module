import flask
import pandas as pd
import numpy as np
import plotly.utils
import json
from scripts.main import define_features, create_json_obj, create_header_and_discription, HAC_rand_score, input_features

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template("index.html")

@app.route('/results')
def clustering():
    return flask.render_template("results.html")


@app.route('/cluster', methods=['POST'])
def cluster():
    # get input
    inputs = [x for x in flask.request.form.values()]
    inputs = str(inputs[0]).split()
    # number of buckets
    n_buckets = int(inputs[0])

    # define features

    temperature, humidity, air_pressure = input_features(inputs)
    # load raw data
    pathes = define_features(temperature = temperature, humidity = humidity, air_pressure = air_pressure)

    #create graph objects for JS
    graphJSON_clusters, graphJSON_temp, graphJSON_hum, graphJSON_pressure = create_json_obj(pathes, n_test_buckets = n_buckets, Test = False, outlier = False)

    #create header and description
    header, description_1, description_2 = create_header_and_discription(n_buckets)

    #metric_string
    #metrics = HAC_rand_score(pathes, n_test_buckets = n_buckets, Test = True)
    return flask.render_template('graph.html',
                                 graphJSON_clusters = graphJSON_clusters,
                                 graphJSON_temperature = graphJSON_temp,
                                 graphJSON_hum = graphJSON_hum,
                                 graphJSON_pressure = graphJSON_pressure,
                                 header = header,
                                 description_1 = description_1,
                                 description_2 = description_2,
                                 #metrics = metrics
                                 )

if __name__ == '__main__':
    app.run(debug=True)