#modules
"""
Modulo para instanciar observaciones, filtrar tendencia y obtener terciles
Dado que los datos de referencia utilizados tiene problemas en la variable temporal quedan cosas
por mejorar, a saber:
Cambiar la coordenada temporal del netcdf para hacerla compatible o poder obtener el pivot year
como atributo con xarray
Esto afecta a la funcion manipular_nc
"""
import datetime
import numpy as np
import xarray as xr
from pathos.multiprocessing import ProcessingPool as Pool
from pandas.tseries.offsets import *
CORES = 9
ruta = '/datos/osman/nmme/monthly/'
hind_length = 28

def manipular_nc(archivo, variable, lat_name, lon_name, lats, latn, lonw, lone,
                 last_month, year_init):
    #hay problemas para decodificar las fechas, genero un xarray con mis fechas decodificadas
    dataset = xr.open_dataset(archivo, decode_times=False)
    var_out = dataset[variable].sel(**{lat_name: slice(lats, latn), lon_name:
                                       slice(lonw, lone)})
    lon = dataset[lon_name].sel(**{lon_name: slice(lonw, lone)})
    lat = dataset[lat_name].sel(**{lat_name: slice(lats, latn)})
    pivot = datetime.datetime(1960, 1, 1) #dificilmente pueda obtener este atributo del nc sin
    #poder decodificarlo con xarray
    time = [pivot + DateOffset(months=int(x), days=15) for x in dataset['T']]
    #genero xarray con estos datos para obtener media estacional
    ds = xr.Dataset({variable: (('time', lat_name, lon_name), var_out)},
                    coords={'time': time, lat_name: lat, lon_name: lon})
    #como el resampling trimestral toma el ultimo mes como parametro
    var_out = ds[variable].resample('Q-' + last_month, dim='time', how='mean')
    #selecciono trimestre de interes
    mes = datetime.datetime.strptime(last_month, '%b').month
    var_out = var_out.sel(time=np.logical_and(var_out['time.month'] == mes,
        np.logical_and(var_out['time.year'] >= year_init,var_out['time.year']
                       <= (year_init+hind_length))))
    return var_out, lat, lon

class Observ(object):
    def __init__(self, institution, var_name, lat_name, lon_name, date_begin,
                 date_end):
        #caracteristicas comunes de todas las observaciones
        self.institution = institution
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.date_begin = date_begin
        self.date_end = date_end

#methods
    def select_months(self, last_month, year_init, lats, latn, lonw, lone):
        """computes seasonal mean"""
        print("seasonal mean")
        file = ruta + self.var_name + '_monthly_nmme_' + self.institution +'.nc'
        [variable, latitudes, longitudes] = manipular_nc(file, self.var_name,
                                                         self.lat_name,
                                                         self.lon_name, lats,
                                                         latn, lonw, lone,
                                                         last_month,
                                                         year_init)
        #converts obs pp unit to (mm/day) in 30-day month type
        variable = np.array(variable)
        if self.var_name == 'prec':
            variable = variable / 30
        return latitudes, longitudes, variable

    def remove_trend(self, observation, CV_opt):
        """removes trend"""
        print("Removing trend")
        [ntimes,nlats,nlons] = observation.shape
        anios = np.arange(ntimes) #en anios es un quilombo y para el caso es lo mismo
        i = np.repeat(np.arange(ntimes ,dtype=int), nlats * nlons)
        j = np.tile(np.repeat(np.arange(nlats, dtype=int), nlons), ntimes)
        k = np.tile(np.arange(nlons, dtype=int), ntimes * nlats)
        p = Pool(CORES)
        p.clear()

        if CV_opt: #validacion cruzada ventana 1 anio
            print("Validacion cruzada")
            CV_matrix = np.logical_not(np.identity(ntimes))

            def filtro_tendencia(i, j, k, anios=anios, CV_m=CV_matrix,
                                 obs=observation): #forecast 4D
                A = np.array([anios[CV_m[:, i]], np.ones((anios.shape[0]-1))])
                d = np.linalg.lstsq(A.T, obs[CV_m[:, i], j, k])[0]
                obs_dt = obs[i, j, k] - (d[0] * anios[i] + d[1])
                return obs_dt

            res = p.map(filtro_tendencia, i.tolist(), j.tolist(), k.tolist())
            observation_dt = np.reshape(np.squeeze(np.stack(res)),
                                        [ntimes, nlats, nlons])
            del(filtro_tendencia, res)
            p.close()

        else:
            def filtro_tendencia(i, j, k, anios=anios, obs=observation): #forecast 4D
                A = np.array([anios, np.ones(anios.shape[0])])
                d = np.linalg.lstsq(A.T, obs[:, j, k])[0]
                obs_dt = obs[i, j, k] - (d[0] * anios[i] + d[1])
                return obs_dt

            res = p.map(filtro_tendencia, i.tolist(), j.tolist(), k.tolist())
            observation_dt = np.reshape(np.squeeze(np.stack(res)),
                                        [ntimes, nlats, nlons])
            del(filtro_tendencia, res)
            p.close()

        return observation_dt

    def computo_terciles(self, observation, CV_opt):
        """obtains terciles limits"""
        print("observed terciles limits")
        ntimes = observation.shape[0]
        if CV_opt: #validacion cruzada ventana 1 anio
            i = np.arange(ntimes)
            p = Pool(CORES)
            p.clear()
            print("Validacion cruzada")
            CV_matrix = np.logical_not(np.identity(ntimes))
            def cal_terciles(i, CV_m=CV_matrix, obs=observation):
                A = np.sort(np.rollaxis(obs[CV_m[:, i], :, :], 0, 3), axis=-1,
                            kind='quicksort')
                upper = A[:, :, np.int(np.round((ntimes - 1) / 3) - 1)]
                lower = A[:, :, np.int(np.round((ntimes - 1) / 3 * 2) - 1)]
                return upper, lower

            res = p.map(cal_terciles, i.tolist())
            terciles = np.stack(res, axis=1)
            del(cal_terciles, res)
            p.close()

        else:
            A = np.sort(np.rollaxis(observation, 0, 3), axis=-1,
                        kind='quicksort')
            upper = A[:, :, np.int(np.round((observation.shape[0]) / 3) - 1)]
            lower = A[:, :, np.int(np.round((observation.shape[0]) / 3 * 2)
                                   - 1)]
            terciles = np.rollaxis(np.stack([upper, lower], axis=2), 2, 0)
        return terciles

    def computo_categoria(self, observation, tercil):
        """assings observed category: Below, normal, Above"""
        print("Observed Category")
        [ntimes, nlats, nlons] = np.shape(observation)
        #calculo el tercil observado cada anio
        obs_terciles = np.empty([3, ntimes, nlats, nlons])
        #below normal
        obs_terciles[0, :, :, :] = observation <= tercil[0, :, :, :]
        #above normal
        obs_terciles[2, :, :, :] = observation >= tercil[1, :, :, :]
        #near normal
        obs_terciles[1, :, :, :] = np.logical_not(np.logical_or(
            obs_terciles[0, :, :, :], obs_terciles[2, :, :, :]))
        return obs_terciles
