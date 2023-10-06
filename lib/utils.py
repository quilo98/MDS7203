
import numpy as np
import cv2
import datetime
import os
import fiona
import rasterio
from rasterio.mask import mask
from glob import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

def crops_satellite_image(path_shp: str, 
                          path_bands: str,
                          path_new_bands: str):
    """
    Function that crops all the images of the bands so that they only consider the client's field.

    Parameters
    ----------
    path_shp : str
        Path where the mask to be used is located.
    path_bands : str
        Path where the images to be cut are located.
    path_new_bands : str
        Path where to save the new images.
    """
    
    with fiona.open(path_shp, "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]
    
    for band in glob(os.path.join(path_bands, '*' + '.tif')):

        with rasterio.open(band) as src:
            out_image, out_transform = mask(src, geoms, invert=False)
            out_meta = src.meta.copy()

        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

        with rasterio.open(path_new_bands + '/' + band[len(path_bands) + 1:-4] + '_crops.tif', "w", **out_meta) as dest:
            dest.write(out_image)
        

def read_band_as_array(path: str):
    """
    A .tiff file is read as a numpy array.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    np.array
        Numpy array with the information of the .tiff file.
    """

    band = rasterio.open(path)
    return band.read(1)


def LST(B4: np.array,
        B5: np.array,
        B10: np.array):
    """
    The LST is calculated with bands 4, 5 and 10.

    Parameters
    ----------
    B4 : np.array
        Band 4 (Red).
    B5 : np.array
        Band 5 (NIR).
    B10 : np.array
        Band 10 (Thermal infrared 1).

    Returns
    -------
    np.array
        Array with the LST.
    """
    
    K2 = 1321.0789
    K1 = 774.8853
    ML = 0.0003342
    AL = 0.1
    NDVIV = 0.5
    NDVIS = 0.2
    
    L_lambda = (ML * B10) + AL
    BT = ((K2) / (np.log(K1 / (L_lambda + 1)))) - 273.15
    NDVI = (B5 - B4) / (B5 + B4)
    PV = np.square((NDVI - NDVIS) / (NDVIV - NDVIS))
    epsilon_lambda = 0.004 * PV + 0.986
    LST = (BT) / (1 + (((0.00115 * BT) / (1.4388)) * np.log(epsilon_lambda)))
    return LST

def RH(TA: list, 
       datetimes: list,  
       B4: np.array,
       B5: np.array,
       B10: np.array,
       B11: np.array,
       H: np.array):
    """
    The relative humidity distribution of a polygon is calculated.

    Parameters
    ----------
    TA : list
        List with ambient temperature map.
    datetimes: list
        List with matrix datetimes for two days.
    B4 : np.array
        Band 4 (Red).
    B5 : np.array
        Band 5 (NIR).
    B10 : np.array
        Band 10 (Thermal infrared 1).
    B11 : np.array
        Band 11 (Thermal infrared 2).
    H : np.array
        Polygon elevations.

    Returns
    -------
    np.array
        Relative humidity distribution of a polygon.
    """
    output = []
    NDVIV = 0.5
    NDVIS = 0.2
    NDVI = (B5 - B4) / (B5 + B4)
    PV = np.square((NDVI - NDVIS) / (NDVIV - NDVIS))

    h, w = NDVI.shape
    epsilo10_epsilon11 = np.ones((h, w), dtype=float) * 0.9939
    num = 0.0
    den = 0.0
    mean_B10 = np.nanmean(B10)
    mean_B11 = np.nanmean(B11)
    for i in range(h):
        for j in range(w):
            num += (B10[i][j] - mean_B10) * (B11[i][j] - mean_B11) 
            den += (B10[i][j] - mean_B10)**2
            if (NDVI[i][j] <= NDVIV) and (NDVI[i][j] >= NDVIS):
                epsilo10_epsilon11[i][j] = ((0.0195 * PV[i][j]) + 0.9688) / ((0.0149 * PV[i][j]) + 0.9747)
            elif NDVI[i][j] > NDVIV:
                epsilo10_epsilon11[i][j] = 0.9966
    
    R_11_10 = num / den
    tau11_tau10 = epsilo10_epsilon11 * R_11_10
    PWV = np.ones((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            if tau11_tau10[i][j] >= 0.9:
                PWV[i][j] = (-18.973 * tau11_tau10[i][j]) + 19.13
            else:
                PWV[i][j] = (-13.412 * tau11_tau10[i][j]) + 14.158

    Q = 0.001 * (12.405 + (1.753 * PWV) + (-0.0762 * np.square(PWV)))
    PA = 1013.3 - (0.1038 * H)
    E = (Q * PA) / 0.622
    for d in range(len(TA)):
        output_d = []
        for m in range(len(TA[d])):
            ES = 611 * np.exp((17.27 * TA[d][m]) / (237.3 + TA[d][m]))
            RH = E / ES
            RH = RH[:,:, np.newaxis]
            DATATIMES = datetimes[d][m][:,:, np.newaxis]
            output_d.append(np.concatenate((RH, DATATIMES), axis=2))
        output.append(np.array(output_d))
    return output

def process_LST_and_measurements(LST: np.array,
                                 measurements: pd.DataFrame,
                                 n: int,
                                 date: str):
    """
    The ambient temperature maps of a day are calculated with the LST and the measurements of that day.

    Parameters
    ----------
    LST : np.array
        Array with the LST.
    measurements : pd.DataFrame
        Dataframe with the measurements of a station for one day.
    n : int
        Number of LST elements to be used to train the linear regression.
    date : str
        Measurement day.

    Returns
    -------
    list
        List of four day's ambient temperature maps.
    """
    
    print('     Se estan preparando los datos para el entrenamiento')
    X_lst = LST[np.logical_not(np.isnan(LST))]
    X = np.array([[0, 0]])
    Y = np.array([0])
    for i in range(len(measurements) - 1):
        lts_r = np.random.choice(X_lst, n, replace=False)
        lts_r = lts_r[:, np.newaxis]
        tem_i = np.full_like(lts_r, measurements['Temperatura Ambiente'].iloc[i])
        add_X = np.concatenate((lts_r, tem_i), axis=1)
        X = np.concatenate((X, add_X))
        Y = np.concatenate((Y, np.ones(n) * measurements['Temperatura Ambiente'].iloc[i + 1]))
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)
    model = LinearRegression()
    print('     Se esta entrenando el modelo')
    model.fit(X, Y)

    times = ['T00:00', 'T00:15', 'T00:30', 'T00:45', 'T01:00', 'T01:15', 'T01:30', 'T01:45', 'T02:00', 'T02:15', 'T02:30', 'T02:45', 
             'T03:00', 'T03:15', 'T03:30', 'T03:45', 'T04:00', 'T04:15', 'T04:30', 'T04:45', 'T05:00', 'T05:15', 'T05:30', 'T05:45', 
             'T06:00', 'T06:15', 'T06:30', 'T06:45', 'T07:00', 'T07:15', 'T07:30', 'T07:45', 'T08:00', 'T08:15', 'T08:30', 'T08:45', 
             'T09:00', 'T09:15', 'T09:30', 'T09:45', 'T10:00', 'T10:15', 'T10:30', 'T10:45', 'T11:00', 'T11:15', 'T11:30', 'T11:45', 
             'T12:00', 'T12:15', 'T12:30', 'T12:45', 'T13:00', 'T13:15', 'T13:30', 'T13:45', 'T14:00', 'T14:15', 'T14:30', 'T14:45', 
             'T15:00', 'T15:15', 'T15:30', 'T15:45', 'T16:00', 'T16:15', 'T16:30', 'T16:45', 'T17:00', 'T17:15', 'T17:30', 'T17:45', 
             'T18:00', 'T18:15', 'T18:30', 'T18:45', 'T19:00', 'T19:15', 'T19:30', 'T19:45', 'T20:00', 'T20:15', 'T20:30', 'T20:45', 
             'T21:00', 'T21:15', 'T21:30', 'T21:45', 'T22:00', 'T22:15', 'T22:30', 'T22:45', 'T23:00', 'T23:15', 'T23:30', 'T23:45', 'T23:59:59']
    output = []
    one_day = datetime.timedelta(days=1)
    for j in range(4):
        output_d = []
        date_start = datetime.date.fromisoformat(date) + (one_day * j)
        date_end = date_start + one_day
        mask = (measurements['ts'] >= date_start.isoformat()) & (measurements['ts'] < date_end.isoformat())
        measurements_mask_1 = measurements.loc[mask]
        avg = np.mean(measurements_mask_1['Temperatura Ambiente'].to_numpy())
        print(f'     Se estan generando los TA del día {j + 1}')
        for i in range(len(times) - 1):
            limit_1 = date_start.isoformat() + times[i]
            limit_2 = date_start.isoformat() + times[i + 1]
            mask = (measurements_mask_1['ts'] >= limit_1) & (measurements_mask_1['ts'] < limit_2)
            measurements_mask_2 = measurements_mask_1.loc[mask]
            print(f'        Mascara para {limit_1} y {limit_2} con {len(measurements_mask_2)} elementos')
            if len(measurements_mask_2) != 0:
                avg = np.mean(measurements_mask_2['Temperatura Ambiente'].to_numpy())

            lst_u = np.unique(LST[~np.isnan(LST)])
            lst_uv = lst_u[:, np.newaxis]
            tem = np.full_like(lst_uv, avg)
            predict = model.predict(np.concatenate((lst_uv, tem), axis=1))
            output_m = np.empty(LST.shape, dtype=float)
            output_m[:] = np.nan
            for k in range(lst_u.shape[0]):
                indexs = np.where(LST==lst_u[k])
                for l in range(indexs[0].shape[0]):
                    output_m[indexs[0][l]][indexs[1][l]] = predict[k]
            output_d.append(output_m)
        output.append(np.array(output_d))
    return output

def process_precipitation_two_dyas(precipitation: pd.DataFrame,
                                   date_start: str):
    """
    The two-day precipitation measurements from a dataframe and from the date delivered are separated.

    Parameters
    ----------
    precipitation : pd.DataFrame
        Dataframe with precipitation measurements.
    date_start : str
        Dat start.

    Returns
    -------
    list
        List with precipitation measurements separated in days.
    """
    
    date_end_1 = datetime.date.fromisoformat(date_start) + datetime.timedelta(days=1)
    date_end_2 = datetime.date.fromisoformat(date_start) + datetime.timedelta(days=2)

    mask = (precipitation['ts'] >= date_start) & (precipitation['ts'] < date_end_1.isoformat())
    precipitation_mask_1 = precipitation.loc[mask]

    mask = (precipitation['ts'] >= date_end_1.isoformat()) & (precipitation['ts'] < date_end_2.isoformat())
    precipitation_mask_2 = precipitation.loc[mask]

    return [precipitation_mask_1['Precipitacion'].to_numpy(), precipitation_mask_2['Precipitacion'].to_numpy()]

def generate_datetime_for_two_days_15_m(date_start: str,
                                        width: int,
                                        height: int):
    """
    The datetime matrices are generated for two days.

    Parameters
    ----------
    date_start : str
        Date start.
    width : int
        Width of matrix datetimes.
    height : int
        Height of matrix datetimes.

    Returns
    -------
    list
        List with matrix datetimes for two days.
    """

    times = ['T00:15', 'T00:30', 'T00:45', 'T01:00', 'T01:15', 'T01:30', 'T01:45', 'T02:00', 'T02:15', 'T02:30', 'T02:45', 
             'T03:00', 'T03:15', 'T03:30', 'T03:45', 'T04:00', 'T04:15', 'T04:30', 'T04:45', 'T05:00', 'T05:15', 'T05:30', 'T05:45', 
             'T06:00', 'T06:15', 'T06:30', 'T06:45', 'T07:00', 'T07:15', 'T07:30', 'T07:45', 'T08:00', 'T08:15', 'T08:30', 'T08:45', 
             'T09:00', 'T09:15', 'T09:30', 'T09:45', 'T10:00', 'T10:15', 'T10:30', 'T10:45', 'T11:00', 'T11:15', 'T11:30', 'T11:45', 
             'T12:00', 'T12:15', 'T12:30', 'T12:45', 'T13:00', 'T13:15', 'T13:30', 'T13:45', 'T14:00', 'T14:15', 'T14:30', 'T14:45', 
             'T15:00', 'T15:15', 'T15:30', 'T15:45', 'T16:00', 'T16:15', 'T16:30', 'T16:45', 'T17:00', 'T17:15', 'T17:30', 'T17:45', 
             'T18:00', 'T18:15', 'T18:30', 'T18:45', 'T19:00', 'T19:15', 'T19:30', 'T19:45', 'T20:00', 'T20:15', 'T20:30', 'T20:45', 
             'T21:00', 'T21:15', 'T21:30', 'T21:45', 'T22:00', 'T22:15', 'T22:30', 'T22:45', 'T23:00', 'T23:15', 'T23:30', 'T23:45', 'T23:59:59']
    day_2 = datetime.datetime.fromisoformat(date_start) + datetime.timedelta(days=1)
    output_d1 = []
    output_d2 = []
    n_like = np.empty((height, width), dtype=object)
    for t in times:
        output_d1.append(np.full_like(n_like, datetime.datetime.fromisoformat(date_start + t)))
        output_d2.append(np.full_like(n_like, datetime.datetime.fromisoformat(day_2.strftime("%Y-%m-%d") + t)))
    return np.array([output_d1, output_d2])

def process_ambient_humidity_botrytis(ambient_humidity: np.array):
    """
    The ambient humidity distribution maps are processed to generate leaf wetness hours.

    Parameters
    ----------
    ambient_humidity : np.array
        Maps with the distribution of the ambient humidity of a place for a whole day.

    Returns
    -------
    np.array
        Distribution of leaf wetness hours of a day for a location.
    """

    output = np.zeros((ambient_humidity[0].shape[0], ambient_humidity[0].shape[1]))
    for i in range(ambient_humidity[0].shape[0]):
        for j in range(ambient_humidity[0].shape[1]):
            h = [k[i][j] for k in ambient_humidity]
            output[i][j] = calculate_leaf_wetness_botrytis(h)
    return output

def calculate_leaf_wetness_botrytis(humidity: list):
    """
    The number of hours of leaf wetness is calculated.

    Parameters
    ----------
    humidity : list
        List with the ambient humidity of a position (x, y) of a day.

    Returns
    -------
    float
        Number of hours with leaf wetness.
    """

    hum_thresh = 98.0
    W = datetime.timedelta(0)
    dt = datetime.timedelta(0)
    four_hrs = datetime.timedelta(hours=4)
    hum_prev = humidity[0][0] * 100
    time_prev = humidity[0][1]
    time_prev_in = humidity[0][1]

    for h in humidity[1:]:
        if (h[0] * 100 > hum_thresh) and (hum_prev > hum_thresh):
            dt = h[1] - time_prev
            time_prev_in = h[1]
            W = W + dt

        dt_in = h[1] - time_prev_in
        if dt_in >= four_hrs:
            W = datetime.timedelta(0)

        hum_prev = h[0] * 100
        time_prev = h[1]

    return W.seconds/3600

def generate_img(map: np.array,
                 levels: dict,
                 path: str):
    """
    An image is generated with the risk levels of the fungus (levels).

    Parameters
    ----------
    map : np.array
        Matrix with risk indexes.
    levels : dict
        Dictionary with the different watering levels of the fungus, with their 
        respective ranges and the associated alert color.
    path : str
        Path where the image will be saved.
    """
    
    img = np.zeros((map.shape[0], map.shape[1], 3)) 
    for row in range(map.shape[0]):
        for column in range(map.shape[1]): 
            for key in levels.keys():
                if map[row][column] > levels[key][0][0] and map[row][column] <= levels[key][0][1]:
                    img[row][column] = levels[key][1]
                    break

    cv2.imwrite(path, img)

def averages_temperatures(temperatures: np.array):
    """
    An array is calculated with the average temperatures of a day.

    Parameters
    ----------
    temperatures : np.array
        Array with temperature arrays.

    Returns
    -------
    np.array
        Array with the average temperatures of a day
    """

    output = temperatures[0] 
    for T in temperatures[1:]:
        output += T
    return output / len(temperatures)

def trasform_temperatures(temperatures: np.array):
    """
    Transform the temperature matrix according to the variables T0, T1 and T2. 
    In addition, it returns the indices of temperatures greater than T2.

    Parameters
    ----------
    temperatures : np.array
        Temperature matrix.

    Returns
    -------
    tuple
        Tuple with the transformed temperature matrix and indexes.
    """

    T0 = 12.0
    T1 = 32.0
    T2 = 40.0
    new_temperatures = np.copy(temperatures)
    new_temperatures[new_temperatures < T0], new_temperatures[np.logical_and(new_temperatures >= T1, new_temperatures <= T2)] = T0, T1

    return new_temperatures, np.where(new_temperatures > T2)

def botrytis_risk_map(temperatures: np.array,
                      leaf_wetness: np.array):
    """
    The risk map for botrytis fungus is calculated with the Broome model.

    Parameters
    ----------
    temperatures : np.array
        Temperature matrix.
    leaf_wetness : np.array
        Leaf wetness matrix.

    Returns
    -------
    np.array
        Matrix with risk indexes.
    """
    
    a0 = -2.647866 
    a1 = -0.374927
    a2 =  0.061601 
    a3 = -0.001511

    temperatures, outliers = trasform_temperatures(temperatures)
    index = a0 + a1*leaf_wetness + a2*leaf_wetness*temperatures + a3*leaf_wetness*temperatures**2

    for i in range(len(outliers[0])):
        index[outliers[0][i]][outliers[1][i]] = 0.0

    return index

def transform_vector(original, new_size):
    """
    An interpolated vector is created with the information of an original vector so that it has the indicated size.

    Parameters
    ----------
    original : _type_
        Original vector.
    new_size : _type_
        Size of new vector.

    Returns
    -------
    list
        New vector.
    """
    
    if len(original) == new_size:
        return original
    else:
        min_val, max_val = original[0], original[-1]
        new_vector = np.linspace(min_val, max_val, num=new_size)

        for i in range(new_size):
            if i == 0:
                new_vector[i] = original[0]
            elif i == new_size-1:
                new_vector[i] = original[-1]
            else:
                x = (i-1)/(new_size-2)*(len(original)-1)
                a, b = int(x), int(x)+1
                frac = x-a
                new_vector[i] = (1-frac)*original[a] + frac*original[b]

        return new_vector

def process_ambient_humidity_oidio(ambient_humidity: np.array,
                                   precipitation: np.array):
    """
    The ambient humidity distribution maps are processed to generate leaf wetness hours.

    Parameters
    ----------
    ambient_humidity : np.array
        Maps with the distribution of the ambient humidity of a place.

    Returns
    -------
    np.array
        Distribution of leaf wetness hours of two days for a location.
    """

    output = np.zeros((ambient_humidity[0].shape[0], ambient_humidity[0].shape[1]))
    for i in range(ambient_humidity[0].shape[0]):
        for j in range(ambient_humidity[0].shape[1]):
            h = [k[i][j] for k in ambient_humidity]
            new_precipitation = transform_vector(precipitation, len(h))
            output[i][j] = calculate_leaf_wetness_oidio(h, new_precipitation)
    return output

def calculate_leaf_wetness_oidio(humidity: list,
                                 precipitation: list): 
    """
    The number of hours of leaf wetness is calculated.

    Parameters
    ----------
    humidity : list
        List with the ambient humidity of a position (x, y) of two days.
    precipitation : list
        List with the ambient precipitation of a station of two days.

    Returns
    -------
    float
        Number of hours with leaf wetness.
    """
    
    hum_thresh = 90.0
    cent_inch = 0.254
    starting = True
    W = datetime.timedelta(0)
    dt = datetime.timedelta(0)
    dt_in = datetime.timedelta(0)
    eight_hrs = datetime.timedelta(hours=8)
    hum_prev = humidity[0][0] * 100
    time_prev = humidity[0][1]
    time_prev_in = humidity[0][1]

    for i in range(1, len(humidity)):
        pre_cond = precipitation[i] >= cent_inch
        if pre_cond:
            starting = False
            starting_date = humidity[i][1]

        if (not starting):
            hum_cond = (humidity[i][0] * 100 >= hum_thresh) and (hum_prev >= hum_thresh)

            if (hum_cond or pre_cond) and (humidity[i][1] >= starting_date):
                dt = humidity[i][1] - time_prev
                time_prev_in = humidity[i][1]
                W = W + dt

            dt_in = humidity[i][1] - time_prev_in
            if dt_in >= eight_hrs:
                W = datetime.timedelta(0)
                dt = datetime.timedelta(0)
                dt_in = datetime.timedelta(0)
                starting = True

        hum_prev = humidity[i][0] * 100
        time_prev = humidity[i][1]

    return W.seconds/3600

def cel2fahr(Cel: float): 
    """
    Celsius degrees are converted to Fahrenheit degrees.

    Parameters
    ----------
    Cel : float
        Degrees celcius.

    Returns
    -------
    float
        Degrees fahrenheit.
    """
    
    Fahr = 9.0/5.0 * Cel + 32
    return Fahr

def req_wet_hrs(temperature: float,
                leaf_wetness: float): 
    """
    The risk of oidio fungus in its first stage is calculated.

    Parameters
    ----------
    temperature : float
        Average temperatura for a position (x,y).
    leaf_wetness : float
        Leaf wetness hours of a day for a position (x,y).

    Returns
    -------
    int
        Binary variable that indicates whether or not risk exists.
    """
    
    Tavg = [42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,75.0,76.0,77.0,78.0]
    Hwet = [40.0,34.0,30.0,27.3,25.3,23.3,20.0,20.0,19.3,18.0,17.3,16.7,16.0,16.0,14.7,14.7,14.0,14.0,13.3,13.3,12.7,12.0,12.0,12.7,14.0,17.3]
    
    Tfah = cel2fahr(temperature)
    Tmin = Tavg[ 0]
    Tmax = Tavg[-1]
    if Tfah < Tmin:
        Tfah = Tmin
    if Tfah > Tmax:
        Tfah = Tmax    
    hrs = np.interp(Tfah, Tavg, Hwet)
    if leaf_wetness/hrs >= 1.0:
        return 1
    else:
        return 0

def oidio_risk_map_phase_one(temperatures: np.array,
                             leaf_wetness: np.array): 
    """
    The risk map for phase one oidio fungus is calculated.

    Parameters
    ----------
    temperatures : np.array
        Temperature matrix.
    leaf_wetness : np.array
        Leaf wetness matrix.

    Returns
    -------
    np.array
        Matrix with risk indexes.
    """

    w, h = temperatures.shape
    index = np.zeros((w, h), dtype=float) 
    for i in range(w):
        for j in range(h):
            if not np.isnan(temperatures[i][j]):
                index[i][j] = req_wet_hrs(temperatures[i][j], leaf_wetness[i][j])
            else:
                index[i][j] = np.nan
    return index

def calculate_oidio_risk_phase_two(days: np.array):
    """
    The oidio risk index is calculated for a position (x, y).

    Parameters
    ----------
    days : np.array
        List with temperature lists of one position every fifteen minutes for the four days.

    Returns
    -------
    float
        Oidio risk index.
    """

    starting = True
    hrs_in_a_row = 6.0
    quarter = 0.25
    threequarter = 0.75
    days_in_a_row = 0
    risk_index = 0

    # Limit temperatures
    T_min = 21.0
    T_max = 30.0
    T_crt = 35.0

    # Temperature counters and associated booleans
    temp_hrs = np.zeros(4, dtype=float)
    incr_cond = False
    decr_cond = False
    temp_cond = np.zeros(4, dtype=bool)
    time_cond = np.zeros(4, dtype=bool)

    for d in days:
        if starting:
            temp_hrs[0] = 0.0
            temp_cond[0] = False
            time_cond[0] = False
            # Loop over one day data-frame
            for t in d:
                temp_cond[0] = (t > T_min) and (t < T_max)
                # Check temperature condition
                if temp_cond[0]:
                    temp_hrs[0] += quarter
                    time_cond[0] = temp_hrs[0] >= hrs_in_a_row
                # Check hrs in a row condition, if yes go to next day
                if time_cond[0]:
                    days_in_a_row += 1
                    risk_index += 20
                    if days_in_a_row == 3:
                        starting = False
                        break                    
                    break
                elif (not time_cond[0]) & (not temp_cond[0]):
                    temp_hrs[0] = 0.0 
        # After 3 days in a row, check day to day conditions
        elif not starting:
            temp_hrs[1] = 0.0
            temp_hrs[2] = 0.0
            temp_hrs[3] = 0.0

            incr_cond = False
            decr_cond = False

            temp_cond[1] = False
            temp_cond[2] = False
            temp_cond[3] = False

            time_cond[1] = False
            time_cond[2] = False
            time_cond[3] = False
            # Loop over one day data-frame
            for t in d:      
                temp_cond[1] = (t >= T_min) & (t <= T_max)
                temp_cond[2] = (not temp_cond[1]) & (t < T_crt)
                temp_cond[3] = t >= T_crt
                # Chek inside time band
                if temp_cond[1]:
                    temp_hrs[1] += quarter
                    time_cond[1] = temp_hrs[1] >= hrs_in_a_row
                # Check outside time band
                elif temp_cond[2]:
                    temp_hrs[1] = 0.0
                    temp_hrs[2] += quarter
                    time_cond[2] = (temp_hrs[2] >= threequarter) & (temp_hrs[2] < hrs_in_a_row) 
                # Check above critical temperature
                elif temp_cond[3]:
                    temp_hrs[3] += quarter
                    time_cond[3] = temp_hrs[3] >= quarter
                # Save increment/decrement conditions
                incr_cond =   incr_cond | time_cond[1]
                decr_cond = ((decr_cond | time_cond[2]) & (not incr_cond)) | time_cond[3]
            # Set risk index
            if incr_cond:
                risk_index += 20
            if decr_cond:
                risk_index -= 10
            risk_index = min(100,max(0,risk_index))
            
        # Check one starting day without time condition
        if starting and (not time_cond[0]) and (days_in_a_row!=3):
            days_in_a_row = 0
            risk_index = 0

    return risk_index

def oidio_risk_map_phase_two(days: np.array):
    """
    The risk map for oidio fungus is calculated with the Gubler model.

    Parameters
    ----------
    days : np.array
        List with temperature maps of one poligone every fifteen minutes for the four days.

    Returns
    -------
    np.array
        Matrix with risk indexes.
    """

    w, h = days[0][0].shape
    index = np.zeros((w, h), dtype=float)
    for i in range(w):
        for j in range(h):
            d = [[t[i][j] for t in days[0]], [t[i][j] for t in days[1]], [t[i][j] for t in days[2]], [t[i][j] for t in days[3]]]
            if d[0][0][np.isnan(d[0][0])] != 0:
                index[i][j] = np.nan
            else:
                index[i][j] = calculate_oidio_risk_phase_two(d)
    return index

def risk_avg(map: np.array):
    """
    The average of the risk matrix is calculated.

    Parameters
    ----------
    map : np.array
        Matrix with risk indexes.

    Returns
    -------
    float
        Average of the matrix with the risk indexes.
    """
    return np.mean(map)
