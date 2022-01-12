import pandas as pd
import statsmodels.api as sm
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
import pickle
import numpy as np
import time
from tslearn.metrics import dtw
from plotly import figure_factory as ff
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fclusterdata
from sklearn.metrics import rand_score
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import json
import plotly
from scipy.cluster import hierarchy
from scipy.spatial.distance import euclidean


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, (time2-time1)))

        return ret
    return wrap

def input_features(inputs):

    if 'temperature' in inputs:
        temperature = True
    else:
        temperature = False

    if 'humidity' in inputs:
        humidity = True
    else:
        humidity = False

    if 'pressure' in inputs:
        pressure = True
    else:
        pressure = False

    if (temperature == False and humidity == False and pressure == False):
        raise ValueError("Non of the features were defined!")

    return temperature, humidity, pressure

#define pathes
def define_features(temperature = True, humidity = False, air_pressure = False, real = True):
    pathes = []

    if real:
        if temperature == True:
            pathes.append('data/10_temp_6days.pkl')
            #pathes.append('data/temp_test.pkl')
        if humidity == True:
            pathes.append('data/10_humidity_6days.pkl')
        if air_pressure == True:
            pathes.append('data/10_pressure_6days.pkl')
        #print(pathes)

    else:
        if temperature == True:
            pathes.append('data/temp_test.pkl')
            #pathes.append('data/temp_test.pkl')
        if humidity == True:
            pathes.append('data/humidity_test.pkl')
        if air_pressure == True:
            pathes.append('data/pressure_test.pkl')
        #print(pathes)

    return pathes

def load_weather_dataset(path):
    ds = pd.read_csv(path)

    #K to C
    ds['temp'] = ds['temp']-273
    ds = ds.iloc[0:8800] #one year of data

    #index
    ds.index = pd.to_datetime(ds['dt_iso'])
    columns = ['temp', 'humidity', 'pressure']

    ds = ds[columns]

    ds['temp'].to_pickle('/Users/e.saurov/PycharmProjects/practical_module/practical_module/clustering/data/temp_test.pkl')
    ds['humidity'].to_pickle(
        '/Users/e.saurov/PycharmProjects/practical_module/practical_module/clustering/data/humidity_test.pkl')
    ds['pressure'].to_pickle(
        '/Users/e.saurov/PycharmProjects/practical_module/practical_module/clustering/data/pressure_test.pkl')
    return print('the data were saved')

pathes = define_features(temperature = False, humidity = False, air_pressure = True)

def get_index(pathes):
    V_tensor = []
    for path in pathes:
        V_matrix = pickle.load(open(path, "rb"))
        V_tensor.append(V_matrix)

    index = V_tensor[0].index
    return index

#create V tensor
def get_mts(pathes, n_test_buckets = 10, scale = True):
    scaler = TimeSeriesScalerMeanVariance()
    V_tensor = []
    for path in pathes:
        V_matrix  = pickle.load(open(path, "rb"))
        V_matrix = np.asarray(V_matrix)

        if scale:
            V_matrix_scaled = scaler.fit_transform(V_matrix)
            V_tensor.append(V_matrix_scaled)
        else:
            V_tensor.append(V_matrix)


    n_observations = V_tensor[0].shape[0]
    n_buckets = V_tensor[0].shape[1]
    length = len(pathes)
    print('defined number of buckets')
    print(n_test_buckets)
    V_tensor = np.asarray(V_tensor)
    V_tensor = V_tensor.reshape(length, n_observations, n_buckets)
    print(V_tensor.shape)
    if V_tensor.shape[2] >= n_test_buckets:
        V_tensor = V_tensor[:, :, 0:n_test_buckets]  # four buckets
    else:
        for number in range(1, n_test_buckets):
            V_tensor = np.concatenate((V_tensor, V_tensor), axis=2)
    n_buckets = V_tensor[0].shape[1]
    print('Tensor was defined')
    print(V_tensor.shape)
    """
    for i in range(0, len(pathes)):
        V_tensor[i] = V_tensor[i].reshape(n_buckets, n_observations)
    """

    return np.asarray(V_tensor)


def get_test_mts(pathes, outlier = False, n_test_buckets = 3, n_clusters = 2, scale = True):
    scaler = TimeSeriesScalerMinMax()
    V_tensor = []
    for path in pathes:
        V_matrix  = pickle.load(open(path, "rb"))
        #print(V_matrix)
        V_matrix = np.asarray(V_matrix)
        print('V_matrix shape: ', V_matrix.shape)
        #print(V_matrix)
        if scale:
            V_matrix_scaled = scaler.fit_transform(V_matrix)
            V_tensor.append(V_matrix_scaled)
        else:
            V_tensor.append(V_matrix)
    print(np.asarray(V_tensor).shape)
    #V_tensor[0] = V_tensor[0].reshape(np.asarray(V_tensor).shape[1], np.asarray(V_tensor).shape[0])
    n_observations = V_tensor[0].shape[0]
    n_buckets = V_tensor[0].shape[1]
    length = len(pathes)
    print('defined number of buckets')
    print(n_test_buckets)
    V_tensor = np.asarray(V_tensor)
    V_tensor = V_tensor.reshape(length, n_observations, n_buckets)
    print(V_tensor.shape)
    if V_tensor.shape[2] >= n_test_buckets:
        V_tensor = V_tensor[:, :, 0:n_test_buckets] #four buckets
    else:
        for number in range(1, n_test_buckets):
            V_tensor = np.concatenate((V_tensor, V_tensor[:, :, 0].reshape(length, n_observations, 1)), axis = 2)
    n_buckets = V_tensor[0].shape[1]
    if n_test_buckets != V_tensor.shape[2]:
        print(V_tensor.shape[2])
        raise ValueError('The synthetic data has incorrect dimension')
    else:
        pass
    print('Tensor was defined')
    print(V_tensor.shape)
    if outlier:
        rix = np.random.randint(0, n_observations/2) #to the random index in the first part of the observations
        V_tensor[0, rix:(rix+3), 1] = 3 #turn three temperature points to outliers

    else:
        #add random data
        l = len(V_tensor[0, :, 0])
        if n_clusters <= 1:
            raise ValueError('No clusters were defined')
        elif n_clusters > n_buckets:
            raise ValueError('More clusters were defined than the number of buckets')
        else:
            for feature in range(0, len(pathes)): #to each feature in pathes
                for i in range(1, n_test_buckets):  # to each bucket except the first one
                    mean = np.random.random(1)
                    sigma = np.random.random(1)/10
                    V_tensor[feature, :, i] = V_tensor[feature, :, 0] + np.random.normal(mean, sigma, l) #add random normal noise with mean 2 and variance 1

    return V_tensor

def p_to_altitude(p):
    k = 44330
    p0 = 101995
    pot = (1 / 5.255)

    altitude = k*(1 - (p/p0)**pot)
    return round(altitude, 0)


def altitudes_of_feature(pathes, n_test_buckets = 10, scale = False):

    V_tensor = get_mts(pathes, n_test_buckets, scale)


    altitudes = []
    means = []
    for bucket in range(0, n_test_buckets):
        mean = np.mean(V_tensor[:, :, bucket])
        means.append(mean)
        altitude = p_to_altitude(mean)
        altitudes.append(altitude)

    return np.asarray(altitudes).reshape(10, 1)

def plot_temperature(pathes, outlier = False):
    V_tensor = get_test_mts(pathes, outlier)

    n_buckets = len(V_tensor[0, 0, :])
    for i in range(0, n_buckets):
        plt.plot(V_tensor[0, :, i])
    return plt.show()

def build_distance_matrix(pathes, n_test_buckets = 3, Test = False, outlier = False):

    if Test:
        tensor = get_test_mts(pathes, outlier, n_test_buckets)
    else:
        tensor = get_mts(pathes, n_test_buckets)
    print('Here is the tensor before building the matrix')
    print(tensor.shape)

    n_features = tensor.shape[0]
    n_observations = tensor.shape[1]
    n_buckets = tensor.shape[2]

    tensor = tensor.reshape(n_features, n_buckets, n_observations)

    """buckets 1, 6, 9"""
    short = False
    if short:
        tensor = np.concatenate((tensor[:, 1, :].reshape(n_features, 1, n_observations),
                                 tensor[:, 6, :].reshape(n_features, 1, n_observations),
                                 tensor[:, 9, :].reshape(n_features, 1, n_observations)),
                                axis = 1)

        print('shortened tensor: ', tensor.shape)

    n_dim = len(tensor[0, :, 0])
    distance_matrix = np.zeros(shape=(n_dim, n_dim))
    for feature in range(0, len(tensor)):
        for i in range(n_dim):
            for j in range(n_dim):
                x = tensor[feature, :, i]
                y = tensor[feature, :, j]
                if i != j:
                    # dist, _ = fastdtw(x, y, dist=euclidean)
                    dist = euclidean(x, y)
                    distance_matrix[i, j] = dist + distance_matrix[i, j]
    print(distance_matrix)
    return distance_matrix

def condense_distance_matrix(pathes, n_test_buckets =3, Test = False, outlier = False):
    distance_matrix = build_distance_matrix(pathes, n_test_buckets, Test, outlier)
    condensed_distance_matrix = squareform(distance_matrix)
    linkresult = linkage(condensed_distance_matrix)
    return linkresult

def local_dendro(pathes):
    altitudes = altitudes_of_feature(pathes, n_test_buckets = 10, scale = False)
    Z = hierarchy.linkage(altitudes, 'single')
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    return plt.show()

# cluster
@timing
def create_fig_for_dendrogram(pathes, n_test_buckets = 3, Test = False, outlier = False):
    if Test:
        V_tensor = get_test_mts(pathes, outlier, n_test_buckets, scale = True)
        n_buckets = n_test_buckets
        #n_buckets = len(V_tensor[0, 0, :])
    else:
        V_tensor = get_mts(pathes, n_test_buckets, scale = False)
        n_buckets = n_test_buckets
        #raise ValueError('The real data were taken')
        #pass

    altitudes = altitudes_of_feature(pathes, n_test_buckets = 10, scale = False)


    distance_matrix = build_distance_matrix(pathes, n_test_buckets, Test, outlier)
    fig = ff.create_dendrogram(altitudes, color_threshold=200)
    fig.update_layout(width=2100,
                      height=700,
                      template=pio.templates['plotly_white'],
                      title="Dendrogram of {} buckets".format(n_test_buckets))

    index = get_index(pathes)

    global fig_temperature
    global fig_humidity
    global fig_pressure

    """temperature"""
    fig_temperature = go.Figure()
    fig_temperature.update_layout(width=2100,
                                  height=700,
                                  template=pio.templates['plotly_white'],
                                  title="Temperature")
    if 'data/10_temp_6days.pkl' in pathes or 'data/temp_test.pkl' in pathes:
        traces = []
        for bucket in range(0, n_buckets):
            trace = go.Scattergl(x=index,
                                 y=V_tensor[0, :, bucket], #temperature
                                 xaxis='x2',
                                 yaxis='y2',
                                 showlegend=True
                                 )
            traces.append(trace)
        fig_temperature.add_traces(traces)
        #fig_temperature.add_traces(traces[1]) #number 1, 6, 9 because they are easily devided into two clusters
        #fig_temperature.add_traces(traces[6])
        #fig_temperature.add_traces(traces[9])
    else:
        pass

    """humidity"""
    fig_humidity = go.Figure()
    fig_humidity.update_layout(width=2100,
                               height=700,
                               template=pio.templates['plotly_white'],
                               title="Humidity")

    print(pathes)
    if 'data/10_humidity_6days.pkl' in pathes or 'data/humidity_test.pkl' in pathes:

        traces_hum = []
        for bucket in range(0, n_buckets):
            trace = go.Scattergl(x=index,
                                 y=V_tensor[1, :, bucket],  # temperature
                                 xaxis='x2',
                                 yaxis='y2',
                                 showlegend=True
                                 )
            traces_hum.append(trace)
        fig_humidity.add_traces(traces_hum)
        #fig_humidity.add_traces(traces_hum[1])  # number 1, 6, 9 because they are easily devided into two clusters
        #fig_humidity.add_traces(traces_hum[6])
        #fig_humidity.add_traces(traces_hum[9])
    else:
        text = 'Fuck, fucke, ucfck, ufcek, fukce, FUCK, kcuf'
        print(text)
        pass


    """air pressure"""
    fig_pressure = go.Figure()
    fig_pressure.update_layout(width=2100,
                               height=700,
                               template=pio.templates['plotly_white'],
                               title="Air pressure")
    if 'data/10_pressure_6days.pkl' in pathes or 'data/pressure_test.pkl' in pathes:
        traces_pressure = []
        for bucket in range(0, n_buckets):
            if 'data/10_humidity_6days.pkl' in pathes or 'data/humidity_test.pkl' in pathes:

                trace = go.Scattergl(x=index,
                                     y=V_tensor[2, :, bucket],  # temperature
                                     xaxis='x2',
                                     yaxis='y2',
                                     showlegend=True
                                     )
            elif 'data/10_temp_6days.pkl' in pathes or 'data/temp_test.pkl' in pathes:
                trace = go.Scattergl(x=index,
                                     y=V_tensor[1, :, bucket],  # temperature
                                     xaxis='x2',
                                     yaxis='y2',
                                     showlegend=True
                                     )
            else:
                trace = go.Scattergl(x=index,
                                     y=V_tensor[0, :, bucket],  # temperature
                                     xaxis='x2',
                                     yaxis='y2',
                                     showlegend=True
                                     )
            traces_pressure.append(trace)
        fig_pressure.add_traces(traces_pressure)
        #fig_pressure.add_traces(traces_pressure[1])  # number 1, 6, 9 because they are easily devided into two clusters
        #fig_pressure.add_traces(traces_pressure[6])
        #fig_pressure.add_traces(traces_pressure[9])

    else:
        text = 'Fuck, fucke, ucfck, ufcek, fukce, FUCK, kcuf'
        print(text)
        pass

    return fig, fig_temperature, fig_humidity, fig_pressure

def create_json_obj(pathes, n_test_buckets =3, Test = False, outlier = False):
    fig, fig_temperature, fig_humidity, fig_pressure = create_fig_for_dendrogram(pathes, n_test_buckets, Test, outlier)
    graphJSON_clusters = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_temp = json.dumps(fig_temperature, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_hum = json.dumps(fig_humidity, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_pressure = json.dumps(fig_pressure, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON_clusters, graphJSON_temp, graphJSON_hum, graphJSON_pressure

def create_header_and_discription(n_buckets):
    header = "Clustering of {} buckets".format(n_buckets)
    description_1 = "The hierarchical clustering algorithm was applied to the dataset consisting of {} buckets and 6 days of observations.".format(n_buckets)
    description_2 = "The DTW distance was used as a metric of similarity between clusters."

    return header, description_1, description_2

def HAC_rand_score(pathes, n_test_buckets = 10, Test = False, outlier = False):
    #inkresult = condense_distance_matrix(pathes, n_test_buckets, Test, outlier)
    linkresult = altitudes_of_feature(pathes)
    print(linkresult)
    labels = fclusterdata(linkresult, t = 3, criterion = 'maxclust', method = 'average')
    target = [1]
    for number in range(0, n_test_buckets-1): #add as many labels as the number of synthetic buckets
        target.append(0)
    print(target)
    print(labels)
    rs = rand_score(target, labels)
    metric_string = ['rand score is: ']
    metric_string.append(rs)
    metric_string.append(str(labels))
    metric_string.append(str(target))
    return print(labels)


if __name__ == '__main__':
    #altitudes_of_feature(pathes, scale=False)
    #local_dendro(pathes)
    HAC_rand_score(pathes)
    #load_weather_dataset('/Users/e.saurov/PycharmProjects/practical_module/practical_module/clustering/data/weather_features.csv')