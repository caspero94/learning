import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import load_model
import os

from funcion import get_data, add_indicator, prepare_data

np.set_printoptions(formatter={'float_kind':'{:f}'.format})
pd.set_option('display.max_columns', None)

'''
symbol = "BTC/BUSD"
timeframe = "4h"
target = 6
loopback = 42
epocas = 10
batch_siz = 8
'''
print("--------------------------------------------------------------")
print("----------  Entrenador de red Neuronal By Caspero  -----------")
print("--------------------------------------------------------------")
print("--------  Seleciona los parametros de entrenamiento  ---------")
print("--------------------------------------------------------------")

# ESTABLECER CUALES VA A SEN LOS PARAMETRO DE ENTRENAMIENTO
symbol = input("--  Símbolo que quiere usar, Ejemplo BTC/BUSD: ")
timeframe = input("--  Timeframe que quiere usar, Ejemplo 4h: ")
indicadores = (input('--  Indicadores disponibles: "sma","ema","dema","wma","rsi","log_return","atr","stoch","macd","bbands","adx","cci","obv","roc": '))
target = int(input("--  Target a predecir, Ejemplo 6: "))
loopback = int(input("--  Loopback aplicado, Ejemplo 42: "))
epocas = int(input("--  Cantidad de épocas, Ejemplo 10: "))
batch_siz = int(input("--  Tamaño del batch, Ejemplo 8: "))

print("--------------------------------------------------------------")
print("-------  Parametros selecionados para entrenamiento  ---------")
print("--------------------------------------------------------------")
print("--",symbol,"-",timeframe,"-",indicadores,"--")
print("--------------------------------------------------------------")
print("-----  Obteniendo datos, indicadores, formateando datos  -----")
# Obtener los datos de entrenamiento, añadir los indicadores, preparar datos
df = get_data(symbol, timeframe)
df = add_indicator(df, *eval(f"[{indicadores}]"))
column_names, scaler, train_x_loopback, train_date, train_x, train_y, test_x_loopback, test_date, test_x, test_y  = prepare_data(df, target, loopback)   

print(train_x)
print(train_x_loopback)
print(train_y)
# caracteristicas totales
t_inputs = train_x.shape[1] * loopback
#"sma:10", "sma:55", "rsi", "log_return", "atr"
print("--------------------------------------------------------------")
print("------------  Nº Caracteristicas como input:",t_inputs,"------------")
print("--------------------------------------------------------------")
print("--------------  Establecer estructura neuronal  --------------")
print("--------------------------------------------------------------")

# Crear el modelo
model = Sequential()
# Pedir la primera capa
capa = input("Opciones para la primera capa (LSTM, GRU, Conv1D): " )
lista_capas = []
if capa == "LSTM":
    units = int(input("Ingresa el número de unidades para esta capa: "))
    model.add(LSTM((units), return_sequences=True,input_shape=(loopback, train_x.shape[1])))
    n_capa = (capa,units)
    lista_capas.append(n_capa)
elif capa == "GRU":
    units = int(input("Ingresa el número de unidades para esta capa: "))
    model.add(GRU((units), return_sequences=True, input_shape=(loopback, train_x.shape[1])))
    n_capa = (capa,units)
    lista_capas.append(n_capa)
elif capa == "Conv1D":
    filtros = int(input("Ingresa el número de filtros para esta capa: "))
    kernel_size = int(input("Ingresa el tamaño del kernel para esta capa: "))
    model.add(Conv1D(filters=filtros, kernel_size=kernel_size, activation='relu', input_shape=(loopback, train_x.shape[1])))
    n_capa = (capa,filtros,kernel_size)
    lista_capas.append(n_capa)
else:
    print("Has introducido mal el nombre de la capa")
# Pedir más capas hasta que el usuario decida parar
while True:
    print("--------------------------------------------------------------")
    capa = input("¿Agregar otra capa? (no, Dropout, Dense, LSTM, GRU, Conv1D): ")
    if capa == "no":
        model.add(Dense(1))
        n_capa = ("Dense", 1)
        lista_capas.append(n_capa)
        break
        
    elif capa == "Dropout":
        porcentaje = float(input("Ingresa el porcentaje de dropout (0-1): "))
        model.add(Dropout(porcentaje))
        n_capa = (capa,porcentaje)
        lista_capas.append(n_capa)
        
    elif capa == "Dense":
        units = int(input("Ingresa el número de unidades para esta capa: "))
        model.add(Dense(units, activation='relu'))
        n_capa = (capa, units)
        lista_capas.append(n_capa)
        
    elif capa == "LSTM":
        units = int(input("Ingresa el número de unidades para esta capa: "))
        model.add(LSTM((units)))
        n_capa = (capa,units)
        lista_capas.append(n_capa)
        
    elif capa == "GRU":
        units = int(input("Ingresa el número de unidades para esta capa: "))
        model.add(GRU((units)))
        n_capa = (capa,units)
        lista_capas.append(n_capa)
        
    elif capa == "Conv1D":
        filtros = int(input("Ingresa el número de filtros para esta capa: "))
        kernel_size = int(input("Ingresa el tamaño del kernel para esta capa: "))
        model.add(Conv1D(filters=filtros, kernel_size=kernel_size, activation='relu'))
        n_capa = (capa,filtros,kernel_size)
        lista_capas.append(n_capa)
        
'''
model.add(LSTM((t_inputs), return_sequences=True,input_shape=(loopback, train_x.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM((t_inputs)))
model.add(Dropout(0.2))
model.add(Dense(t_inputs, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
'''
# Compilar el modelo
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mse')

print("--------------------------------------------------------------")
print("--------------   Estructura neuronal definida  ---------------")
print("--------------------------------------------------------------")
print("-- Target:", target, "- Loopback:", loopback, "- Epochs:", epocas, "- Batch:", batch_siz, " --")
print("--------------------------------------------------------------")
print("---------------------  Capas definidas  ----------------------")
print("--",lista_capas,"--")
print("--------------------------------------------------------------")
print("-----------------  Comenzando entrenamiento  -----------------")
print("--------------------------------------------------------------")
# Entrenar el modelo
history = model.fit(train_x_loopback, train_y, epochs=epocas, batch_size=batch_siz, validation_data=(test_x_loopback, test_y), verbose=1)
print("--------------------------------------------------------------")
print("-----------------  Entrenamiento finalizado  -----------------")
print("--------------------------------------------------------------")


# Especicar ruta models, y nombre modelo
ruta_models = "models"
nombre_modelo = f"{symbol}-{timeframe}-{indicadores}-target{target}-loopback{loopback}-epochs{epocas}-batch{batch_siz}.h5"
nombre_modelo = nombre_modelo.replace("/", "-").replace('"', "").replace(",", "-").replace(":", "-")
ruta_archivo = os.path.join(ruta_models, nombre_modelo)

# Guardar modelo
model.save(ruta_archivo)
print("Guardado: " + nombre_modelo)


print("--------------------------------------------------------------")
print("---------------  Lista de modelos disponibles  ---------------")
lista_models = os.listdir(ruta_models)  # Listar archivos
for model in lista_models:
    print(model)
print("--------------------------------------------------------------")
nombre_modelo = input("-- Escribe el modelo que quieres usar: ")
ruta_archivo = os.path.join(ruta_models, nombre_modelo)
# Cargar modelo
model = load_model(ruta_archivo)
print("--------------------------------------------------------------")
print("Cargando: " + nombre_modelo)

# Hacer predicciones
print("--------------------------------------------------------------")
print("------------------  Comenzando prediciones  ------------------")
print("--------------------------------------------------------------")
train_predict = model.predict(train_x_loopback)
test_predict = model.predict(test_x_loopback)
print("--------------------------------------------------------------")
print("-----------------  Prediciones finalizadas  ------------------")
print("--------------------------------------------------------------")
# Imprimir todas las dimensiones de las variables usadas
train_predict.shape
'''
shapes = [train_x_loopback, train_date, train_x, train_y, test_x_loopback, test_date, test_x, test_y, train_predict, test_predict]
variables = locals()

for x in shapes:
    # busca el nombre de la variable correspondiente
    var_name = [name for name in variables if variables[name] is x][0]
    # imprime el nombre y la forma de la variable
    print(var_name + " dimension: " + str(x.shape))
    
    

print("Train_x_LOOPBACK")
print(train_x_loopback[0])
print("Train_x")
print(train_x[0:loopback])
'''

# Separar nombres columnas
datetime_column_name = column_names[0]
values_column_names = column_names[1:]

# DATAFRAME PREDICION

# Unir los arrays de entrenamiento y prueba
x_train_combined = np.concatenate((train_x, train_predict), axis=1)
x_test_combined = np.concatenate((test_x, test_predict), axis=1)

# Invertir normalizacion
train_predict = scaler.inverse_transform(x_train_combined)
test_predict = scaler.inverse_transform(x_test_combined)

# Convertir en dataframe unido, los valores con lo predicho
train_df = pd.DataFrame(train_predict,columns=values_column_names)
test_df = pd.DataFrame(test_predict,columns=values_column_names)

# Agregar fechas al dataframe de prediciones
train_df = pd.concat([train_date, train_df], axis=1)
test_df = pd.concat([test_date, test_df], axis=1)

# Unir train y test
df_predic = pd.concat([train_df, test_df], ignore_index=True)

# DATAFRAME REAL

# Unir los arrays de entrenamiento y prueba
x_train_real = np.concatenate((train_x, train_y), axis=1)
x_test_real = np.concatenate((test_x, test_y), axis=1)

# Invertir normalizacion
train_real = scaler.inverse_transform(x_train_real)
test_real = scaler.inverse_transform(x_test_real)

# Convertir en dataframe unido, los valores con lo predicho
train_df_real = pd.DataFrame(train_real,columns=values_column_names)
test_df_real = pd.DataFrame(test_real,columns=values_column_names)

# Agregar fechas al dataframe de prediciones
train_df_real = pd.concat([train_date, train_df_real], axis=1)
test_df_real = pd.concat([test_date, test_df_real], axis=1)

# Unir train y test
df_real = pd.concat([train_df_real, test_df_real], ignore_index=True)

# CREAMOS DATAFRAME COMPARANDO PREDICION Y REAL

df_compare = pd.merge(df_real[["datetime","target"]], df_predic[["datetime","target"]], on='datetime')
df_compare = df_compare.rename(columns={'target_x': 'target_real', 'target_y': 'target_predic'})
df_compare['target_real'] = df_compare['target_real'].astype(int)
df_compare['target_predic'] = df_compare['target_predic'].astype(int)
df_compare = df_compare.set_index('datetime')

# GRAFICAMOS LOS RESULTADOS
n_velas = int(input("Ingresa el número velas a graficar: "))
print("--------------------------------------------------------------")
fig, ax = plt.subplots(figsize=(10, 6))
df_compare.plot(y=['target_real', 'target_predic'], ax=ax)
plt.show()

last_candles = df_compare.tail(n_velas)  # Filtra los últimos 30 días
fig, ax = plt.subplots(figsize=(10, 6))
last_candles.plot(y=['target_real', 'target_predic'], ax=ax)
plt.show()
print("-----  Graficos generados con los resultados obtenidos  ------")
print("--------------------------------------------------------------")