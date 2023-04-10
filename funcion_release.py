import pymongo
import pandas as pd
import pandas_ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(formatter={'float_kind':'{:f}'.format})
pd.set_option('display.max_columns', None)

def get_data(symbol,timeframe):
        
    # Configuramos parametros de obtención datos 
    symbol = symbol
    timeframe = timeframe
    selcol = (symbol+"-"+timeframe)
    
    # Configuramos acceso a datos
    username = "casper"
    password = "caspero"
    cluster = "ClusterCrypto"
    client = pymongo.MongoClient(f"mongodb+srv://{username}:{password}@{cluster}.6ydpkxh.mongodb.net/?retryWrites=true&w=majority")
    db = client["CryptoData"]
    collection = db[selcol]
    
    # Generamos datos de entrenamiento
    df = pd.DataFrame(list(collection.find()))
    df['datetime'] = pd.to_datetime(df['_id'], unit='ms') - pd.Timedelta(hours=5)
    df["volume"] = df["volume"].astype(float)
    df = df[['datetime','open','high','low','close','volume']]
    df = df.sort_values(by="datetime")
    df = df.reset_index(drop=True)
    df = df.drop(columns="open")
    df = df.dropna()
    
    return df

def add_indicator(df, *args):
    for arg in args:
        if isinstance(arg, str):
            arg_parts = arg.split(":")
            if arg_parts[0] == "sma":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 55
                df.ta.sma(append=True, length=length)
            elif arg_parts[0] == "ema":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 55
                df.ta.ema(append=True, length=length)
            elif arg_parts[0] == "log_return":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 16
                df.ta.log_return(append=True, length=length)
            elif arg_parts[0] == "rsi":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 14
                df.ta.rsi(append=True, length=length)
            elif arg_parts[0] == "atr":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 14
                df.ta.atr(append=True, length=length)
            elif arg_parts[0] == "stoch":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 14
                df.ta.stoch(append=True, length=length)
            elif arg_parts[0] == "macd":
                fast = int(arg_parts[1]) if len(arg_parts) > 1 else 12
                slow = int(arg_parts[2]) if len(arg_parts) > 2 else 26
                df.ta.macd(append=True, fast=fast, slow=slow)
            elif arg_parts[0] == "bbands":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 20
                std = int(arg_parts[2]) if len(arg_parts) > 2 else 2
                df.ta.bbands(append=True, length=length, std=std)
            elif arg_parts[0] == "adx":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 14
                df.ta.adx(append=True, length=length)
            elif arg_parts[0] == "cci":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 20
                df.ta.cci(append=True, length=length)
            elif arg_parts[0] == "dema":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 10
                df.ta.dema(append=True, length=length)
            elif arg_parts[0] == "obv":
                df.ta.obv(append=True)
            elif arg_parts[0] == "roc":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 12
                df.ta.roc(append=True, length=length)
            elif arg_parts[0] == "wma":
                length = int(arg_parts[1]) if len(arg_parts) > 1 else 9
                df.ta.wma(append=True, length=length)
            else:
                raise ValueError(f"Unknown indicator: {arg}")
        else:
            raise ValueError(f"Unknown indicator: {arg}")
            
    df.dropna(inplace=True)
    return df.copy()

def prepare_data(df,target,loopback):
    
    df = df
    target = target
    loopback = loopback
    
    # Agregamos el objetivo a predecir
    df['target'] = df['close'].shift(target)
    df = df.dropna()
    column_names = df.columns.tolist()
    df = df.reset_index(drop=True)
    # Separamos los datos a normalizar del datetime y definimos el train_size
    train_size = int(len(df) * 0.8)
    df_date = df["datetime"]
    df_values = df.iloc[:, 1:]
    
    # Normalizamos train y test
    scaler = MinMaxScaler()
    df_values = scaler.fit_transform(df_values)
    
    # Dividir los datos en entrenamiento y test
    train_date = df_date[:train_size]
    train_values = df_values[:train_size]
    train_x = train_values[:, :-1] # Todas las columnas menos la ultima
    train_y = train_values[:, -1:] # Solo la ultima columna
    
    test_date = df_date[train_size:]
    test_values = df_values[train_size:]
    test_x = test_values[:, :-1] # Todas las columnas menos la ultima
    test_y = test_values[:, -1:] # Solo la ultima columna
    
    # Aplicar loopback
    train_x_loopback = np.zeros((train_x.shape[0] - loopback, loopback, train_x.shape[1]))
    for i in range(train_x.shape[0] - loopback):
        train_x_loopback[i, :, :] = train_x[i:i+loopback, :]
        
    test_x_loopback = np.zeros((test_x.shape[0] - loopback, loopback, test_x.shape[1]))
    for i in range(test_x.shape[0] - loopback):
        test_x_loopback[i, :, :] = test_x[i:i+loopback, :]
     
    # Ajustamos los demas datos aplicando el loopback  
    
    train_date = train_date.iloc[:-loopback]
    train_x = train_x[:-loopback, :]
    train_y = train_y[:-loopback, :]
    
    test_date = test_date.iloc[:-loopback]
    test_date = test_date.reset_index(drop=True)
    test_x = test_x[:-loopback, :]
    test_y = test_y[:-loopback, :]

    return column_names, scaler, train_x_loopback, train_date, train_x, train_y, test_x_loopback, test_date, test_x, test_y 

#symbol = "BTC/BUSD"
#timeframe = "1w"
#target = 1

#df = get_data(symbol, timeframe)
#df, column_names, scaler, train_date, train_x, train_y, test_date, test_x, test_y = prepare_data(df, target)   

#def restore_data():

# MOSTRAR TODAS LAS FILAS/COLUMNAS
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# LLAMAR A LA FUNCION GET_DATA
#symbol = "BTC/BUSD"
#timeframe = "1h"
#datos = get_data(symbol, timeframe)
#print(datos.head())

# LLAMAR A LA FUNCION ADD_INDICATOR
#datos = add_indicator(datos, "sma", "rsi","sma:20","bbands:28","macd:5:10")
#print(datos.head())
