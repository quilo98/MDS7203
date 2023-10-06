import lib.utils as utils
import numpy as np
import time
import pandas as pd
#from rasterio.mask import mask

station_id = 54
parameters_ids = [222, 223, 262]
factors = [0.01, 0.01, 0.254]
date_start = '2021-03-17'
date_end = '2021-03-20'

kmzfile = f'./ejemplo_data/kmz/{station_id}.kmz'
shpfile = f'./ejemplo_data/shp/{station_id}/{station_id}.shp'
bands = f'./ejemplo_data/results/{station_id}/bands'
crops_bands = f'./ejemplo_data/results/{station_id}/bands/crops'
dataframe_path = f'./ejemplo_data/results/{station_id}/dataframes'
arrays_path = f'./ejemplo_data/results/{station_id}/arrays'

T0 = time.time()

print('Se estan cortando las imagenes satelitales')
utils.crops_satellite_image(shpfile, bands, crops_bands)

print('Se estan leyendondo las bandas satelitales')
B4 = utils.read_band_as_array(crops_bands + '/B4_crops.tif')
B5 = utils.read_band_as_array(crops_bands + '/B5_crops.tif')
B10 = utils.read_band_as_array(crops_bands + '/B10_crops.tif')
B11 = utils.read_band_as_array(crops_bands + '/B11_crops.tif')
H = utils.read_band_as_array(crops_bands + '/elevation_crops.tif')

print('Se estan extrayendo las mediciones de la estación')
parameters = utils.extract_measurements(station_id, parameters_ids, factors, date_start, date_end)
utils.save_measurements(parameters, dataframe_path)

print('Se estan leyendondo los dataframes') 
# los df deben ser de la forma ts, parámetro, donde parámetro es uno de: Temperatura Ambiente, Precipitacion, Humedad Ambiente
T = pd.read_csv(dataframe_path + '/Temperatura Ambiente.csv')

print('Se esta calculando la LST')
LST = utils.LST(B4, B5, B10)
np.save(arrays_path + '/LST.npy', LST)

h, w = LST.shape
datetimes = utils.generate_datetime_for_two_days_15_m(date_start, w, h)

print('Se estan calculando las TA')
TA = utils.process_LST_and_measurements(LST, T, 10000, date_start) 
np.save(arrays_path + '/TA_96h.npy', np.array(TA))

print('Se estan calculando las RH')
RH = utils.RH(TA[:2], datetimes, B4, B5, B10, B11, H)
np.save(arrays_path + '/RH_48h.npy', np.array(RH))
print(np.array(RH).shape)

T1 = time.time()
print(f'Finalizo. Tiempo total: {T1 - T0} [s]')