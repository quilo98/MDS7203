import lib.utils as utils
import numpy as np
import pandas as pd

levels_phase_one = {'No hay riesgo': [(-9999, 0), [0,255,0]],
                    'Alto riesgo': [(1.0, 9999), [255,0,0]]}

levels_phase_two = {'No hay riesgo': [(-9999, 0), [0,255,0]],
                    'Bajo riesgo': [(0, 30), [255,255,0]],
                    'Riesgo moderado': [(30, 50), [255,128,0]],
                    'Alto riesgo': [(50, 100), [255,0,0]]}

station_id = 54
date_start = '2021-03-17'
path_out_map = f'./data/results/{station_id}/maps/oidio'
path_ambient_humidity_48h = f'./data/results/{station_id}/arrays/RH_48h.npy'
path_temperatures_96h = f'./data/results/{station_id}/arrays/TA_96h.npy'
path_precipitation_48h = f'./data/results/{station_id}/dataframes/Precipitacion.csv'

print('Se estan leyendo los arreglos')
ambient_humidity_48h = np.load(path_ambient_humidity_48h, allow_pickle=True)
temperatures_96h = np.load(path_temperatures_96h)
precipitation_48h = utils.process_precipitation_two_dyas(pd.read_csv(path_precipitation_48h), date_start)

print('Se esta calculando la matriz promedio de la temperatura')
avg_temperatures_24h = utils.averages_temperatures(temperatures_96h[0])

print('Se esta calculando el riesgo de Oidio fase 1')
leaf_wetness_48h = utils.process_ambient_humidity_oidio(np.concatenate((ambient_humidity_48h[0], ambient_humidity_48h[1])), 
                                                        np.concatenate((precipitation_48h[0], precipitation_48h[1])))
risk_map_phase_one = utils.oidio_risk_map_phase_one(avg_temperatures_24h, leaf_wetness_48h)

print('Se esta generando imagen')
utils.generate_img(risk_map_phase_one, levels_phase_one, path_out_map + '_phase_one.png')

print('Se esta calculando el riesgo de Oidio fase 2')
risk_map_phase_two = utils.oidio_risk_map_phase_two(temperatures_96h)

print('Se esta generando imagen')
utils.generate_img(risk_map_phase_two, levels_phase_two, path_out_map + '_phase_two.png')
