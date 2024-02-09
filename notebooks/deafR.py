from scipy import stats
import joblib
from sklearn.linear_model import LinearRegression

#for google drive
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#gauth = GoogleAuth()
#gauth.LocalWebserverAuth()  # Opens a local web server for authentication
#drive = GoogleDrive(gauth)

#file_id = "1ndTa_rwxSZY-ssVz7Ew9htuYSTsfCsty"
#file = drive.CreateFile({'id': file_id})
#file.GetContentFile('Merged_Data.csv', mimetype='text/csv')

##import libraries
from flask import Flask #Mono
from flask import request #Mono

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import lightgbm as lgb
import warnings
import pandas as pd
import numpy as np
import ta
import time
import pickle
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning)

working_dir = "C:/Users/david/OneDrive/Documents/deaf_reload/"

start_time = datetime.now();
last_time = start_time
print('Starting predict: ', start_time)
global origin_data, model2, model3, meta_model

### Load the saved models from the .pkl files
model1 = pickle.load(open(working_dir + "/models/cb_model_log_target.sav", 'rb'))
model2 = pickle.load(open(working_dir + "/models/xg_model_log_target.sav","rb"))
model3 = pickle.load(open(working_dir + "/models/lg_model_log_target.sav","rb"))
model4 = pickle.load(open(working_dir + "/models/hist_model_log_target.sav","rb"))
model5 = pickle.load(open(working_dir + "/models/extree_model_log_target.sav","rb"))
#p95
model6 = pickle.load(open(working_dir + "/models/knn_model_log_target.sav","rb"))
meta_model = pickle.load(open(working_dir + "/models/meta_9model_log_target.sav","rb"))

###import data
origin_data = pd.read_csv('C:/Users/david/OneDrive/Documents/deaf_reload/data/ExportedHistoricSecBarData 2023-01-18.csv',low_memory=False,delimiter=';',header=None, nrows=5000)
#print("data loaded")

#rename columns
origin_data.rename(columns={0: "timestamp_column", 1: "open", 2: "high", 3: "low", 4: "close", 5: "ticks", 6: "spread", 7: "direction", 8: "topWick", 9: "body",10: "bottomWick"}, inplace=True)

#convert datatime to_datetime
origin_data['timestamp_column'] = pd.to_datetime(origin_data['timestamp_column'])
#print(origin_data.head(5))

#drop nan if is any
df_nn = origin_data.dropna()

#create variable of data name to only chance one time
data_variable = df_nn.copy()
print(data_variable.head(5))

##add lags
one_min = -60
two_min = -120
three_min = -180
four_min = -240
five_min = -300
ten_min = -600
fifteen_min = -900
twenty_min = -1200
twentyfive_min = -1500
thirty_min = -1800
fourty_min = -2400
sixty_min = -3600

#features
data_variable['month'] = data_variable['timestamp_column'].apply(lambda x: x.month)
data_variable['day'] = data_variable['timestamp_column'].apply(lambda x: x.day)
data_variable['hour'] = data_variable['timestamp_column'].apply(lambda x: x.hour)
data_variable['body_3'] = data_variable['body'].shift(three_min)
data_variable['body_5'] = data_variable['body'].shift(five_min)
data_variable['body_20'] = data_variable['body'].shift(twenty_min)
data_variable['ticks_2'] = data_variable['ticks'].shift(two_min)
data_variable['ticks_4'] = data_variable['ticks'].shift(four_min)
data_variable['ticks_5'] = data_variable['ticks'].shift(five_min)
data_variable['ticks_15'] = data_variable['ticks'].shift(fifteen_min)
data_variable['ticks_60'] = data_variable['ticks'].shift(sixty_min)
data_variable['spread_5'] = data_variable['spread'].shift(five_min)
data_variable['spread_log'] = np.log(data_variable.iloc[:, 6] + 1)
data_variable['body_log'] = np.log(data_variable.iloc[:, 9] +1)
data_variable['ticks_log'] = np.log(data_variable.iloc[:, 5] +1)
data_variable['spread_sqrt'] = np.sqrt(data_variable.iloc[:, 6])
data_variable['body_sqrt'] = np.sqrt(data_variable.iloc[:, 9])

##features
selected_feature_names = [
    'topWick',
    'body',
    'bottomWick',
    'spread',
    'hour',
    'day',
    'month',
    'body_3',
    'body_5',
    'body_20',
    'ticks_2',
    'ticks_4',
    'ticks_5',
    'ticks_15',
    'ticks_60',
    'spread_5',
    'spread_log',
    'body_log',
    'ticks_log',
    'spread_sqrt',
    'body_sqrt',
]

#drop nan
#print(data_variable.head(5))
df = data_variable.dropna()
#print(df.head(1))

###predictions
X_new_0 = df[selected_feature_names]
print(X_new_0.head(1))
##pick last row
X_new = X_new_0.iloc[:1]
X_new = X_new_0.tail(1)
print(X_new)
print(X_new.columns.to_list())
#predict
pred_model1 = model1.predict(X_new)
pred_model2 = model2.predict(X_new)
pred_model3=  model3.predict(X_new)
pred_model4 = model4.predict(X_new)
pred_model5 = model5.predict(X_new)
pred_model6 = model6.predict(X_new)

### Combine the predictions into a single array
base_model_preds = [pred_model1,pred_model2, pred_model3,pred_model4,pred_model5,pred_model6]
base_model_preds = np.array(base_model_preds).T
print(base_model_preds)

### Make prediction with the meta-model
pred_meta_model = meta_model.predict(base_model_preds)
#print(pred_meta_model,"Predicted by meta model all decimals!!!")
tmp_time = datetime.now()
print('pred_meta_model finished: ', tmp_time - last_time)
last_time = tmp_time

#decimal spaces
rounded_arr = np.round(pred_meta_model, 2)

#Print the final prediction from the meta-model
print(rounded_arr,"Predicted by meta model rounded!!!")