#modules
import numpy as np
import numpy.matlib
import netCDF4
import xarray as xr
from scipy import stats
from scipy import signal
from pathos.multiprocessing import ProcessingPool as Pool

cores = 6

#get netdf variables
#def manipular_nc(archivo,variable,lat_name,lon_name):
#    dataset = netCDF4.Dataset(archivo, 'r')
#    var_out = np.squeeze(dataset.variables[variable][:])
#    lon = dataset.variables[lat_name][:]
#    lat = dataset.variables[lon_name][:]
#    dataset.close()
#    return var_out, lon, lat

def manipular_nc(archivo,variable,lat_name,lon_name,lats, latn, lonw, lone):

    dataset = xr.open_dataset(archivo,decode_times = False)

    var_out = dataset[variable].sel(**{lat_name: slice(lats,latn), lon_name: slice(lonw,lone)})

    lon = dataset[lon_name].sel(**{lon_name: slice(lonw,lone)})

    lat = dataset[lat_name].sel(**{lat_name: slice(lats,latn)})

    return var_out, lat, lon

class Observ(object):
#caracteristicas comunes de todas las observaciones
    def __init__(self, name, institution, var_name, lat_name, lon_name, date_begin, date_end):
        self.name = name
        self.institution = institution
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.date_begin = date_begin
        self.date_end = date_end

        pass

#comienzo la definición de los métodos

#selecciono los datos a partir del mes de IC y plazo, saco promedio trimestral
    def select_months(self, init_cond, target, lats, latn, lonw, lone): #init_cond en meses y target en meses ()
       
        first_month = init_cond + target-1+12*(self.date_begin-1982)
	
        ruta = '/datos/osman/nmme/monthly/'        

        file = ruta + 'precip_monthly_nmme_' + self.institution +'.nc'

        hind_len = (self.date_end-self.date_begin)*12+first_month  #esto hay que emprolijarlo
        
        [variable,latitudes,longitudes] = manipular_nc (file,self.var_name,self.lat_name,self.lon_name,lats,latn,lonw,lone)

        #esto se puede mejorar usando las propiedades de xarray. tengo la cabeza
        #quemada

        variable = np.squeeze(np.array(variable))

        observation = np.nanmean(np.stack((variable[first_month:hind_len:12,:,:],variable[first_month+1:hind_len:12,:,:],variable[first_month+2:hind_len:12,:,:]),axis = 3), axis = 3)
       
        return latitudes, longitudes, observation

#remuevo la tendencia del conjunto de datos: lo hago en crossvalidation con ventana de 3 anios
    def remove_trend(self, observation,CV_opt):  #obs 3-D array array ntimes nlats nlons CV_opt boolean

        [ntimes,nlats,nlons] = observation.shape
        
        anios = np.arange(1982,2011)

        i = np.repeat(np.arange(ntimes,dtype = int),nlats*nlons)

        j = np.tile(np.repeat(np.arange(nlats,dtype = int),nlons),ntimes)

        k = np.tile(np.arange(nlons,dtype = int),ntimes*nlats)

        p = Pool(cores)

        p.clear()

        if CV_opt: #validacion cruzada ventana 1 anio

            print("Validacion cruzada")

            CV_matrix = np.logical_not(np.identity(ntimes))

            def filtro_tendencia(i,j,k,anios=anios,CV_m = CV_matrix,obs=observation): #forecast 4D

                A = np.array([anios[CV_m[:,i]], np.ones((anios.shape[0]-1))])

                d = np.linalg.lstsq(A.T,obs[CV_m[:,i],j,k])[0]

                obs_dt = obs[i,j,k]-(d[0]*anios[i]+d[1])

                return obs_dt

            res = p.map(filtro_tendencia,i.tolist(),j.tolist(),k.tolist())

            observation_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes,nlats,nlons])

            print(observation_dt.shape)

            del(filtro_tendencia,res)

            p.close()

        else:

            def filtro_tendencia(i,j,k,anios=anios,obs=observation): #forecast 4D

                A = np.array([anios, np.ones(anios.shape[0])])

                d = np.linalg.lstsq(A.T,obs[:,j,k])[0]

                obs_dt = obs[i,j,k]-(d[0]*anios[i]+d[1])

                return obs_dt

            res = p.map(filtro_tendencia,i.tolist(),j.tolist(),k.tolist())

            observation_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes,nlats,nlons])

            print(observation_dt.shape)

            del(filtro_tendencia,res)

            p.close()

        return observation_dt

