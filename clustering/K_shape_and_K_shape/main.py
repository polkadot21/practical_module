import time
import numpy as np
import pickle
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape, TimeSeriesKMeans
import matplotlib.pyplot as plt


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, (time2 - time1)))

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

    if temperature == False & humidity == False & pressure == False:
        raise ValueError("Non of the features were defined!")

    return temperature, humidity, pressure


# define pathes
def define_features(temperature=True, humidity=True, air_pressure=True, real=True):
    pathes = []

    if real:
        if temperature == True:
            pathes.append('data/10_temp_6days.pkl')
            # pathes.append('data/temp_test.pkl')
        if humidity == True:
            pathes.append('data/10_humidity_6days.pkl')
        if air_pressure == True:
            pathes.append('data/10_pressure_6days.pkl')
        # print(pathes)

    else:
        if temperature == True:
            pathes.append('data/temp_test.pkl')
            # pathes.append('data/temp_test.pkl')
        if humidity == True:
            pathes.append('data/humidity_test.pkl')
        if air_pressure == True:
            pathes.append('data/pressure_test.pkl')
        # print(pathes)

    return pathes


# create V tensor
def get_mts(pathes, n_test_buckets=10, scale=True, short=True):
    global V_tensor_short
    scaler = TimeSeriesScalerMeanVariance()
    V_tensor = []
    for path in pathes:
        V_matrix = pickle.load(open(path, "rb"))
        V_matrix = np.asarray(V_matrix)

        if scale:
            V_matrix_scaled = scaler.fit_transform(V_matrix)
            V_tensor.append(V_matrix_scaled)
        else:
            V_tensor.append(V_matrix)

    n_observations = V_tensor[0].shape[0]
    n_buckets = V_tensor[0].shape[1]
    length = len(pathes)
    # print('defined number of buckets')
    # print(n_test_buckets)
    V_tensor = np.asarray(V_tensor)
    V_tensor = V_tensor.reshape(length, n_observations, n_buckets)
    # print(V_tensor.shape)
    if V_tensor.shape[2] >= n_test_buckets:
        V_tensor = V_tensor[:, :, 0:n_test_buckets]  # four buckets
    else:
        for number in range(1, n_test_buckets):
            V_tensor = np.concatenate((V_tensor, V_tensor), axis=2)
    n_buckets = V_tensor[0].shape[1]
    # print('Tensor was defined')
    # print(V_tensor.shape)
    """
    for i in range(0, len(pathes)):
        V_tensor[i] = V_tensor[i].reshape(n_buckets, n_observations)
    """

    ###
    if short:
        V_tensor = np.asarray(V_tensor)
        V_tensor_3 = V_tensor[:, :, 3].reshape(3, 8468, 1)
        V_tensor_6 = V_tensor[:, :, 6].reshape(3, 8468, 1)
        V_tensor_9 = V_tensor[:, :, 9].reshape(3, 8468, 1)

        V_tensor_short = np.concatenate((V_tensor_3, V_tensor_6, V_tensor_9), axis=1)

        n_features = 3
        length = 8468
        n_buckets = 3
        V_tensor_short = V_tensor_short.reshape(n_buckets, length, n_features)

    return V_tensor_short


@timing
def kShapeClustering(pathes):
    V_tensor = get_mts(pathes)
    X = V_tensor[:, :, :]
    X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    print(X[1])
    ks = KShape(n_clusters=2, n_init=1, random_state=0).fit_predict(X)

    return ks


def kMeansClustering(pathes):
    V_tensor = get_mts(pathes)
    X = V_tensor[:, :, :]
    # print(X[1])
    ks = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=5, max_iter_barycenter=5, random_state=0).fit_predict(X)

    return ks


def plot_mts(pathes, n_test_buckets=10, scale=False):
    V_tensor = get_mts(pathes, n_test_buckets, scale)
    # print(V_tensor)
    plt.plot(V_tensor[0, :1000, 0])
    # plt.plot(V_tensor[1, :1000, :])
    # plt.plot(V_tensor[2, :1000, :])

    return plt.show()


if __name__ == "__main__":
    pathes = define_features()
    V_tensor = get_mts(pathes)
    print(V_tensor.shape)
    ks = kMeansClustering(pathes)
    print(ks)
    plot_mts(pathes)
