##import libraries
from flask import Flask #Mono
from flask import request #Mono
#from sklearn.experimental import enable_hist_gradient_boosting  # Required for HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
# Import XGBoost
from xgboost import XGBRegressor
# Import CatBoost
from catboost import CatBoostRegressor

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import lightgbm as lgb
from scipy import stats
import warnings
import ta
import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning)

app = Flask(__name__) #Mono

working_dir = "C:/Users/david/OneDrive/Documents/deaf_reload"

#loading historic data
#historic_data = pd.read_csv(working_dir + '/data/ExportedHistoricSecBarData 2023-01-24_small.csv', low_memory=False, delimiter=';', header=None,) 
##rename columns
#historic_data.rename(columns={0: "timestamp_column", 1: "open", 2: "high", 3: "low", 4: "close", 5: "ticks", 6: "spread", 7: "direction", 8: "topWick", 9: "body",10: "bottomWick"}, inplace=True)
##convert datatime to_datetime
#historic_data['timestamp_column'] = pd.to_datetime(historic_data['timestamp_column'])
#print("historic_data rows: ", len(historic_data))
#origin_data = historic_data

### Load the saved models from the .pkl files
model1 = pickle.load(open(working_dir + "/models/cb_model_log_target.sav", 'rb'))
model2 = pickle.load(open(working_dir + "/models/xg_model_log_target.sav","rb"))
model3 = pickle.load(open(working_dir + "/models/lg_model_log_target.sav","rb"))
model4 = pickle.load(open(working_dir + "/models/hist_model_log_target.sav","rb"))
#p95
model5 = pickle.load(open(working_dir + "/models/lg95_model_log_target.sav","rb"))
model6 = pickle.load(open(working_dir + "/models/hist_model95_log_target.sav","rb"))
model7 = pickle.load(open(working_dir + "/models/extree95_model_log_target.sav","rb"))
meta_model = pickle.load(open(working_dir + "/models/meta_7_models_log_target.sav","rb"))
print("Models loaded successfully!")

 ###### Begin Mono
@app.route('/predict', methods=['GET','POST'])
def predict():
    start_time = datetime.now();
    last_time = start_time
    print('Starting predict: ', start_time)
    global origin_data, model2, model3, meta_model

    ##read data
    #origin_df_ = pd.read_csv('C:/Users/mauri/Desktop/deaf_bot/bot_plus_deaf/data/test_last.csv',
    #                         low_memory=False,delimiter=',',skiprows=[0], header= None)

    ##convert datatime to_datetime
    #origin_df['date_time'] = pd.to_datetime(origin_df['datetime'])

    ##columns_names
    #origin_df = origin_df_.rename(columns={0: "datetime", 1: "open", 2: "high", 3: "low", 
    #                                       4: "close", 5: "ticks", 6: "spread"})

    data = {}
    if request.method == 'POST':
        data = {
        "timestamp_column": [request.form['datetime']],
        "open": [float(request.form['open'])],
        "high": [float(request.form['high'])],
        "low": [float(request.form['low'])],
        "close": [float(request.form['close'])],
        "ticks": [float(request.form['ticks'])],
        "spread": [float(request.form['spread'])],
        "direction": [str(request.form['direction'])],
        "topWick": [float(request.form['topWick'])],
        "body": [float(request.form['body'])],
        "bottomWick": [float(request.form['bottomWick'])],
        "body_3": [float(request.form['body_3'])],
        "body_5": [float(request.form['body_5'])],
        "body_20": [float(request.form['body_20'])],
        "ticks_2": [float(request.form['ticks_2'])],
        "ticks_4": [float(request.form['ticks_4'])],
        "ticks_5": [float(request.form['ticks_5'])],
        "ticks_15": [float(request.form['ticks_15'])],
        "ticks_60": [float(request.form['ticks_60'])],
        "spread_5": [float(request.form['spread_5'])]
        }
    else:
        data = {
        "timestamp_column": [request.args.get('datetime')],
        "open": [float(request.args.get('open'))],
        "high": [float(request.args.get('high'))],
        "low": [float(request.args.get('low'))],
        "close": [float(request.args.get('close'))],
        "ticks": [float(request.args.get('ticks'))],
        "spread": [float(request.args.get('spread'))],
        "direction": [str(request.args.get('direction'))],
        "topWick": [float(request.args.get('topWick'))],
        "body": [float(request.args.get('body'))],
        "bottomWick": [float(request.args.get('bottomWick'))],
        "body_3": [float(request.args.get('body_3'))],
        "body_5": [float(request.args.get('body_5'))],
        "body_20": [float(request.args.get('body_20'))],
        "ticks_2": [float(request.args.get('ticks_2'))],
        "ticks_4": [float(request.args.get('ticks_4'))],
        "ticks_5": [float(request.args.get('ticks_5'))],
        "ticks_15": [float(request.args.get('ticks_15'))],
        "ticks_60": [float(request.args.get('ticks_60'))],
        "spread_5": [float(request.args.get('spread_5'))]
        }

    #print("before: ", data)
    data['timestamp_column'] = pd.to_datetime(data['timestamp_column'])
    #print("after: ", data)
    origin_data = pd.DataFrame(data)
    
    #print("datetime: ", data['datetime'])
    #print("datetime converted: ", data['datetime'])
    #origin_data = historic_data[(historic_data['timestamp_column'] < data['datetime'][0])]
    #print("origin_data rows: ", origin_data.shape)

    ###import data
    #origin_data = pd.read_csv('C:/Users/mauri/Desktop/deaf_reload/ExportedSecBarData.csv',low_memory=False,delimiter=';',header=None,)
    #update_data = pd.read_csv('C:/Users/david/AppData/Roaming/MetaQuotes/Tester/9EB2973C469D24060397BB5158EA73A5/Agent-127.0.0.1-3000/MQL5/Files/ExportedSecBarData.csv',low_memory=False,delimiter=';',header=None,)

    #tmp_time = datetime.now()
    #print('update_data loaded: ', tmp_time - last_time)
    #last_time = tmp_time

    ##rename columns
    #origin_data.rename(columns={0: "timestamp_column", 1: "open", 2: "high", 3: "low", 4: "close", 5: "ticks", 6: "spread", 7: "direction", 8: "topWick", 9: "body",10: "bottomWick"}, inplace=True)
    #origin_data['timestamp_column'] = pd.to_datetime(origin_data['timestamp_column'])

    ##convert datatime to_datetime
    #update_data['timestamp_column'] = pd.to_datetime(update_data['timestamp_column'])
    ##print(origin_data)

    ##concatenating historic and origin data
    #print("update_data rows: ", len(update_data))
    #origin_data = pd.concat([ origin_data, update_data ], axis=0, ignore_index=True)
    #print("contactenated rows: ", len(origin_data))
    #tmp_time = datetime.now()
    #print('contactenated loaded: ', tmp_time - last_time)
    #last_time = tmp_time

    #drop nan if is any
    df_nn = origin_data.dropna()

    #create variable of data name to only chance one time
    data_variable = df_nn.copy()


    #features
    data_variable['month'] = data_variable['timestamp_column'].apply(lambda x: x.month)
    data_variable['day'] = data_variable['timestamp_column'].apply(lambda x: x.day)
    data_variable['hour'] = data_variable['timestamp_column'].apply(lambda x: x.hour)
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
    df = data_variable.dropna()

    ###predictions
    X_new_0 = df[selected_feature_names]

    ###pick last row
    #X_new = X_new_0.iloc[:1]
    X_new = X_new_0.tail(1)
    #print(X_new)
    #print(X_new.columns.to_list())

    ##predict
    #tmp_time = datetime.now()
    pred_model1 = model1.predict(X_new)
    pred_model2 = model2.predict(X_new)
    pred_model3=  model3.predict(X_new)
    pred_model4 = model4.predict(X_new)
    pred_model5 = model5.predict(X_new)
    pred_model6 = model6.predict(X_new)
    pred_model7 = model7.predict(X_new)
    
    
    #tmp_time = datetime.now()
    #print('pred_model2 finished: ', tmp_time - last_time)
    #last_time = tmp_time
    #tmp_time = datetime.now()
    #print('pred_model3 finished: ', tmp_time - last_time)
    #last_time = tmp_time

    ### Combine the predictions into a single array
    base_model_preds = [pred_model1,pred_model2, pred_model3,pred_model4,pred_model5,pred_model6,pred_model7]
    base_model_preds = np.array(base_model_preds).T
    #print(base_model_preds)

    ### Make prediction with the meta-model
    pred_meta_model = meta_model.predict(base_model_preds)
    #print(pred_meta_model,"Predicted by meta model all decimals!!!")
    #tmp_time = datetime.now()
    #print('pred_meta_model finished: ', tmp_time - last_time)
    #last_time = tmp_time

    #decimal spaces
    rounded_arr = np.round(pred_meta_model, 2)

    #Print the final prediction from the meta-model
    print(rounded_arr,"Predicted by meta model rounded!!!")

    return str(rounded_arr[0])
# End Mono
