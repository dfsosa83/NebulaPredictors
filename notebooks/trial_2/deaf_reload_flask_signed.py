##import libraries
from flask import Flask #Mono
from flask import request #Mono
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

working_dir = "C:/Users/david/OneDrive/Documents/NebulaPredictors"

### Load the saved models from the .pkl files
model1 = pickle.load(open(working_dir + "/trained_models_trial2/xg_model_signed_target_Q1.sav","rb"))
model2 = pickle.load(open(working_dir + "/trained_models_trial2/cb_model_signed_target_Q2.sav", 'rb'))
model3 = pickle.load(open(working_dir + "/trained_models_trial2/lg_model_signed_target_Q3.sav","rb"))
model4 = pickle.load(open(working_dir + "/trained_models_trial2/hist_model_signed_target_Q4.sav","rb"))
meta_model = pickle.load(open(working_dir + "/trained_models_trial2/meta_hist_4_models_signed_target_Q1_to_Q4.sav","rb"))
print("Models loaded successfully!")

 ###### Begin Mono
@app.route('/predict', methods=['GET','POST'])
def predict():
    start_time = datetime.now();
    last_time = start_time
    print('Starting predict: ', start_time)
    global origin_data, model2, model3, meta_model

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

    ### Combine the predictions into a single array
    base_model_preds = [pred_model1,pred_model2, pred_model3,pred_model4]
    base_model_preds = np.array(base_model_preds).T
    #print(base_model_preds)

    ### Make prediction with the meta-model
    pred_meta_model = meta_model.predict(base_model_preds)

    #decimal spaces
    rounded_arr = np.round(pred_meta_model, 2)

    #Print the final prediction from the meta-model
    print(rounded_arr,"Predicted by meta model rounded!!!")

    return str(rounded_arr[0])
# End Mono
