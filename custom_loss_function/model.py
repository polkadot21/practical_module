import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

#import loss
from dilate_loss import dilate_loss
#import optimizers
from Optimization_with_dilate import Optimization_with_dilate
from Optimization_with_MSE import Optimization_with_MSE
#import models
from LSTMModel import LSTMModel
from GRUModel import GRUModel
from FCNN import FCNN
from seq2seq import EncoderRNN, DecoderRNN, Net_GRU
#import config
import json
#import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import holidays
#import data

from datahub import get_ticker_price, get_BTC_data, get_weather, add_trend_cycle, is_holiday, add_holiday_col
#import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tslearn.metrics import dtw
from scipy.spatial.distance import directed_hausdorff
#import visualisation
from plot_results import plot_predictions
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


with open('config.json') as f:
    config = json.load(f)

#local configs
ticker = "GOOGL"
from_when = "2016-05-01"
now = datetime.datetime.now().strftime("%Y-%m-%d")
#features = ["marketcap(USD)", "adjustedTxVolume(USD)", 'generatedCoins', 'value', 'exchangeVolume(USD)', 'medianTxValue(USD)']
#features = ['Open', 'High', 'Low', 'value', 'Volume',]
features = ['value', 'temp_max', 'pressure']

###less features 2-3 features/specify the amount of test data

data = get_weather(features)
#data = get_BTC_data(features, from_when)

#data = get_ticker_price(ticker, from_when, now, features)
#print(data.columns)
#print(data)


#generate lagged data
def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 7):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

input_dim = 1
#df_generated = generate_time_lags(data, input_dim)

#add holidays
#us_holidays = holidays.US()
#data = add_holiday_col(data, us_holidays)

#add filtering
#data = add_trend_cycle(data)



#columns
print(len(data.columns))




#train/val/test split
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data, 'value', 0.2)

#print(len(X_train.columns))


scaler = MinMaxScaler()

X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)


#prepare tensors
batch_size = config['training']['batch_size']

#get data loaders

def get_loaders(X_train_arr,
                X_val_arr,
                X_test_arr,
                y_train_arr,
                y_val_arr,
                y_test_arr):

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, test_loader_one

train_loader, val_loader, test_loader, test_loader_one = get_loaders(X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#choose a model
def get_model(model, model_params):
    models = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "FCNN":FCNN,
    }
    return models.get(model.lower())(**model_params)

alpha = config["loss"]['alpha']
gamma = config["loss"]['gamma']

config_encoder = {
    "input_dim": len(data.columns)-1,
    "hidden_dim": 128,
    "layer_dim": 1,
    "dropout_prob": 0.2,
    'batch_size': 5,
    }

config_decoder = {
    "input_dim": len(data.columns)-1,
    "hidden_dim": 128,
    "layer_dim": 1,
    "fc_units" : 128,
    "dropout_prob": 0.2,
    "output_dim": 1
    }

seq2seq_config = {
    'encoder' : EncoderRNN(**config_encoder),
    'decoder' : DecoderRNN(**config_decoder),
    'input_dim':1,
    'target_length' : 1,
    "device":'cpu'
    }

print(len(y_test))
print(len(test_loader))
#model seq2seq
model = Net_GRU(**seq2seq_config)
#model =EncoderRNN(**config_encoder)
loss_fn = dilate_loss
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

opt = Optimization_with_dilate(model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer
                  )
opt.train(train_loader,
          val_loader,
          batch_size=config['training']['batch_size'],
          n_epochs=config['training']['n_epochs'],
          n_features=config['model']['input_dim']
          )
#opt.plot_losses()

predictions_seq2seq, values_seq2seq = opt.evaluate(test_loader,
                                   batch_size=5,
                                   n_features = config['model']['input_dim']
                                   )


#model dilate
model = get_model('lstm', config['model'])
loss_fn = dilate_loss
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

opt = Optimization_with_dilate(model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer
                  )
opt.train(train_loader,
          val_loader,
          batch_size=config['training']['batch_size'],
          n_epochs=config['training']['n_epochs'],
          n_features=config['model']['input_dim']
          )
#opt.plot_losses()

predictions_dilate, values = opt.evaluate(test_loader_one,
                                   batch_size=1,
                                   n_features = config['model']['input_dim']
                                   )


#model mse
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
opt = Optimization_with_MSE(model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer
                  )

opt.train(train_loader,
          val_loader,
          batch_size=config['training']['batch_size'],
          n_epochs=config['training']['n_epochs'],
          n_features=config['model']['input_dim']
          )
#opt.plot_losses()

predictions_MSE, values = opt.evaluate(test_loader_one,
                                   batch_size=1,
                                   n_features = config['model']['input_dim']
                                   )

#model FCNN
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
opt = Optimization_with_MSE(model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer
                  )

opt.train(train_loader,
          val_loader,
          batch_size=config['training']['batch_size'],
          n_epochs=config['training']['n_epochs'],
          n_features=config['model']['input_dim']
          )
#opt.plot_losses()

prediction_FCNN, values = opt.evaluate(test_loader_one,
                                   batch_size=1,
                                   n_features = config['model']['input_dim']
                                   )


#predict

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions_dilate, predictions_MSE, prediction_FCNN,  values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds_dilate = np.concatenate(predictions_dilate, axis=0).ravel()
    preds_MSE = np.concatenate(predictions_MSE, axis=0).ravel()
    preds_FCNN = np.concatenate(prediction_FCNN, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": vals,
                                   "prediction_dilate": preds_dilate,
                                   "prediction_MSE": preds_MSE,
                                   'prediction_FCNN':preds_FCNN,
                                   }, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction_dilate", "prediction_MSE", "prediction_FCNN"]])

    return df_result

preds_seq2seq = np.concatenate(predictions_seq2seq, axis=0).ravel()
vals_seq2seq = np.concatenate(values_seq2seq, axis=0).ravel()
df_seq2seq = pd.DataFrame(data = {"value":vals_seq2seq, 'predictions_seq2seq':preds_seq2seq}, index=X_test.head(len(vals_seq2seq)).index)
df_seq2seq = df_seq2seq.sort_index()
df_seq2seq = inverse_transform(scaler, df_seq2seq, [["value", 'predictions_seq2seq']])
seq2seq = {'mae_seq2seq': mean_absolute_error(df_seq2seq.value, df_seq2seq.predictions_seq2seq),
              'rmse_seq2seq': mean_squared_error(df_seq2seq.value, df_seq2seq.predictions_seq2seq) ** 0.5,
              "dtw_seq2seq": dtw(df_seq2seq.value, df_seq2seq.predictions_seq2seq),
              "Symmetric Hausdorff seq2seq": max(
                  directed_hausdorff(
                      np.array(df_seq2seq.value).reshape(-1, 1),
                      np.array(df_seq2seq.predictions_seq2seq).reshape(-1, 1))[0],
                  directed_hausdorff(
                      np.array(df_seq2seq.predictions_seq2seq).reshape(-1, 1),
                      np.array(df_seq2seq.value).reshape(-1, 1))[0]), }
#print(seq2seq)


df_result = format_predictions(predictions_dilate, predictions_MSE, prediction_FCNN, values, X_test, scaler)
#print(df_result)

def calculate_metrics(df):
    #seq2seq = {'mae_dilate': mean_absolute_error(df.value, df.predictions_seq2seq),
     #         'rmse_dilate': mean_squared_error(df.value, df.predictions_seq2seq) ** 0.5,
      #        "dtw_dilate": dtw(df.value, df.predictions_seq2seq),
       #       "Symmetric Hausdorff dilate": max(
        #          directed_hausdorff(
         #             np.array(df.value).reshape(-1, 1),
          #            np.array(df.predictions_seq2seq).reshape(-1, 1))[0],
           #       directed_hausdorff(
            #          np.array(df.predictions_seq2seq).reshape(-1, 1),
             #         np.array(df.value).reshape(-1, 1))[0]), }

    dilate = {'mae_dilate' : mean_absolute_error(df.value, df.prediction_dilate),
            'rmse_dilate' : mean_squared_error(df.value, df.prediction_dilate) ** 0.5,
            "dtw_dilate" : dtw(df.value, df.prediction_dilate),
            "Symmetric Hausdorff dilate" : max(
                directed_hausdorff(
                    np.array(df.value).reshape(-1, 1),
                    np.array(df.prediction_dilate).reshape(-1, 1))[0],
                directed_hausdorff(
                    np.array(df.prediction_dilate).reshape(-1, 1),
                    np.array(df.value).reshape(-1, 1))[0]),}

    mse  = {'mae_MSE': mean_absolute_error(df.value, df.prediction_MSE),
            'rmse_MSE': mean_squared_error(df.value, df.prediction_MSE) ** 0.5,
            "dtw_MSE": dtw(df.value, df.prediction_MSE),
            "Symmetric Hausdorff MSE": max(
                directed_hausdorff(
                    np.array(df.value).reshape(-1, 1),
                    np.array(df.prediction_MSE).reshape(-1, 1))[0],
                directed_hausdorff(
                    np.array(df.prediction_MSE).reshape(-1, 1),
                    np.array(df.value).reshape(-1, 1))[0]),
            }
    fcnn = {'mae_MSE': mean_absolute_error(df.value, df.prediction_FCNN),
            'rmse_MSE': mean_squared_error(df.value, df.prediction_FCNN) ** 0.5,
            "dtw_MSE": dtw(df.value, df.prediction_FCNN),
            "Symmetric Hausdorff MSE": max(
                directed_hausdorff(
                    np.array(df.value).reshape(-1, 1),
                    np.array(df.prediction_FCNN).reshape(-1, 1))[0],
                directed_hausdorff(
                    np.array(df.prediction_FCNN).reshape(-1, 1),
                    np.array(df.value).reshape(-1, 1))[0]),
            }

    return  dilate, mse, fcnn

dilate, mse, fcnn   = calculate_metrics(df_result)



#def build_baseline_model(df, test_ratio, target_col):
 #   X, y = feature_label_split(df, target_col)
  #  X_train, X_test, y_train, y_test = train_test_split(
   #     X, y, test_size=test_ratio, shuffle=False
    #)
    #model = LinearRegression()
    #model.fit(X_train, y_train)
    #prediction = model.predict(X_test)

    #result = pd.DataFrame(y_test)
    #result["prediction"] = prediction
    #result = result.sort_index()

    #return result

#df_baseline = build_baseline_model(data, 0.2, 'value')
#aseline_metrics = calculate_metrics(df_baseline)



if __name__=="__main__":
    print("seq2seq model's metrics:")
    print(seq2seq)
    print('')
    print("Dilate model's metrics:")
    print(dilate)
    print('')
    print("MSE model's metrics:")
    print(mse)
    print('')
    print("FCNN model's metrics:")
    print(fcnn)
    plot_predictions(df_result, df_seq2seq, ticker)
