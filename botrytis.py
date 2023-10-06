import lib.utils as utils
import numpy as np

levels = {'No hay riesgo': [(-9999, 0), [0,255,0]],
          'Bajo riesgo': [(0, 0.5), [255,255,0]],
          'Riesgo moderado': [(0.5, 1.0), [255,128,0]],
          'Alto riesgo': [(1.0, 9999), [255,0,0]]}

station_id = 54
path_out_map = f'./ejemplo_data/results/{station_id}/maps/botrytis.png'
path_ambient_humidity_48h = f'./ejemplo_data/results/{station_id}/arrays/RH_48h.npy'
path_temperatures_96h = f'./ejemplo:data/results/{station_id}/arrays/TA_96h.npy'

print('Se estan leyendo los arreglos')
ambient_humidity_48h = np.load(path_ambient_humidity_48h, allow_pickle=True)
temperatures_96h = np.load(path_temperatures_96h)

print('Se esta calculando la matriz promedio de la temperatura')
avg_temperatures = utils.averages_temperatures(temperatures_96h[0])

print('Se esta calculando el riesgo de Botrytis')
leaf_wetness = utils.process_ambient_humidity_botrytis(ambient_humidity_48h[0])
risk_map = utils.botrytis_risk_map(avg_temperatures, leaf_wetness)

print('Se esta generando imagen')
utils.generate_img(risk_map, levels, path_out_map)
