import time,os,re,csv,sys,uuid,joblib
import pickle
from datetime import date
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from logger import update_predict_log, update_train_log
from read_json_files import fetch_data
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, timedelta

## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", "models")):
    os.mkdir("models") 

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "TS Analysis Capstone"
SAVED_MODEL = os.path.join("models", "model-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION))))


def days_between(d1, d2):
    print("days_between dates:",d1, d2)
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def load_data():
    
    return fetch_data('data\cs-train')

def model_train(df=None, test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file
    Note that the latest training data is always saved to be used by perfromance monitoring tools.
    """

    ## start timer for runtime
    time_start = time.time()
    
    if df is None:
        df = load_data()
        
    ts = df.sort_values(by="invoice_date")
    ts = ts.groupby("invoice_date")["price"].sum()
    model = ARIMA(ts, order=(8, 0, 8)) 
    results_ARIMA = model.fit(disp=-1)

    if test:
        print("... saving test version of model")
        joblib.dump(results_ARIMA, os.path.join("models", "test.joblib"))
    else:
        print("... saving model: {}".format(SAVED_MODEL))
        joblib.dump(results_ARIMA, SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models", 'latest-train.pickle')
        with open(data_file, 'wb') as tmp:
            pickle.dump({'df':df}, tmp)
        
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    update_train_log(len(ts), 'eval_test TBD', runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=test)

def model_predict(date, country, df=None, model=None, test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()
    print("model_predicted started", time_start)
    ## input checks
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise Exception("ERROR (model_predict) - invalid input date {} was given".format(date))
        
    if isinstance(country, str):
        pass
    else:
        raise Exception("ERROR (model_predict) - invalid input country {} was given".format(country))

    if df is None:
        df = load_data()
        
    ## make prediction and gather data for log entry
    if (country != "all"):
        ts = df[df["country"] == country].sort_values(by="invoice_date")
    else:
        ts = df.sort_values(by="invoice_date")

    ts = ts.groupby("invoice_date")["price"].sum()

    nsteps = days_between(str(ts[:-2:-1].keys()[0].date()), date)

    model = ARIMA(ts, order=(8, 0, 8)) 
    results_ARIMA = model.fit(disp=-1)

    predicted = results_ARIMA.predict(start=len(ts), end=len(ts) + nsteps, exog=None, typ='linear', dynamic=False)
    rangeDates = np.array([ts[:-2:-1].keys()[0].date() + timedelta(x) for x in range(nsteps + 1)],dtype='datetime64[D]')
    predicted = pd.Series(predicted.values,rangeDates)
    y_proba = 'None'
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    for i in range(len(predicted)):
        update_predict_log(predicted[i], date, country, 
                           runtime, MODEL_VERSION, test=test)
        
    return({'predicted':predicted})


def model_load(test=False):
    """
    example funtion to load model
    """
    if test : 
        print( "... loading test version of model" )
        model = joblib.load(os.path.join("models","test.joblib"))
        return(model)

    if not os.path.exists(SAVED_MODEL):
        exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
        raise Exception(exc)
    
    model = joblib.load(SAVED_MODEL)
    return(model)

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    ## data ingestion
    #df = load_data()

    ## train the model
    #model_train(df=df, test=True)

    ## load the model
    #model = model_load(test=True)
    
    ## example predict
    country = "united_states"
    date = '2019-07-31'

    result = model_predict(date=date, country=country, df=None, model=None, test=True)
    y_pred = result['predicted']
    print("predicted: {}".format(y_pred))