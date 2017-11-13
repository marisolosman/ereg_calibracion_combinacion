#modules
import numpy as np
import numpy.matlib
import netCDF4
import xarray as xr
from scipy import stats
from scipy import signal
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
cores = 6

#get netdf variables

def manipular_nc(archivo,variable,lat_name,lon_name,lats, latn, lonw, lone):

    dataset = xr.open_dataset(archivo,decode_times = False)

    var_out = dataset[variable].sel(**{lat_name: slice(lats,latn), lon_name: slice(lonw,lone)})

    lon = dataset[lon_name].sel(**{lon_name: slice(lonw,lone)})
    
    lat = dataset[lat_name].sel(**{lat_name: slice(lats,latn)})

    return var_out, lat, lon


class Model(object):
#caracteristicas comunes de todos los modelos
    def __init__(self, name, institution, var_name, lat_name, lon_name, miembros_ensamble, leadtimes, hind_begin, hind_end):
        self.name = name
        self.institution = institution
        self.ensembles = miembros_ensamble
        self.leadtimes = leadtimes
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.hind_begin = hind_begin
        self.hind_end = hind_end

        pass

#imprimir caracteristicas generales del modelo
    def __str__(self):
        return "%s is a model from %s and has %s ensemble members and %s leadtimes" % (self.name, self.institution, self.ensembles, self.leadtimes)

#comienzo la definición de los métodos

#selecciono los datos a partir del mes de IC y plazo, saco promedio trimestral
    def select_months(self, init_cond, target, lats, latn, lonw, lone):  #init_cond en meses y target en meses ()
       
        final_month = init_cond + 11
        if final_month > 12:
            flag_end = 1
            final_month = final_month - 12
        else:
            flag_end = 0
	
        ruta = '/datos/osman/nmme/monthly/'        

        file = ruta + 'prec_Amon_' + self.institution + '-' + self.name + '_'+'19820101_r1i1p1_'+'198201-198212.nc'

        
        hindcast_length = self.hind_end - self.hind_begin 

        [variable,latitudes,longitudes] = manipular_nc (file,self.var_name,self.lat_name,self.lon_name,lats,latn,lonw,lone)

        forecast = np.empty([hindcast_length,self.ensembles,latitudes.shape[0],longitudes.shape[0]])

        #loop sobre los anios del hindcast period
        for i in np.arange(self.hind_begin,self.hind_end):
            for j in np.arange(1,self.ensembles+1):

                
                file = ruta + 'prec_Amon_' + self.institution + '-' + self.name + '_'+str(i)+'{:02d}'.format(init_cond)+'01_r'+str(j)+'i1p1_'+str(i)+'{:02d}'.format(init_cond)+'-'+str(i+flag_end)+'{:02d}'.format(final_month)+'.nc'

                [variable,latitudess,longitudes] = manipular_nc (file,self.var_name,self.lat_name,self.lon_name,lats,latn,lonw,lone)

                              
                forecast[i-self.hind_begin,j-1,:,:] = np.nanmean(np.squeeze(np.array(variable))[target:target+2,:,:], axis = 0)
                variable = []
	# Return values of interest: latitudes longitudes forecast
        return latitudes, longitudes, forecast

#remuevo la tendencia del conjunto de datos: lo hago en crossvalidation con ventana de 3 anios
    def remove_trend(self, forecast, CV_opt):  #forecast 4-D array ntimes nensembles nlats nlons CV_opt boolean

        [ntimes,nmembers,nlats,nlons] = forecast.shape

        anios = np.arange(1982,2011)

        i = np.repeat(np.arange(ntimes,dtype = int),nmembers*nlats*nlons)

        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),nlats*nlons),ntimes)

        j = np.tile(np.repeat(np.arange(nlats,dtype = int),nlons),ntimes*nmembers)

        k = np.tile(np.arange(nlons,dtype = int),ntimes*nmembers*nlats)

        p = Pool(cores)

        p.clear()

        if CV_opt: #validacion cruzada ventana 1 anio

            print("Validacion cruzada")

            CV_matrix = np.logical_not(np.identity(ntimes))

            def filtro_tendencia(i,l,j,k,anios=anios,CV_m = CV_matrix,forec=forecast): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)

                A = np.array([ anios[CV_m[:,i]], np.ones((anios.shape[0]-1))])

                d = np.linalg.lstsq(A.T,y[CV_m[:,i]])[0]

                for_dt = forec[i,l,j,k]-(d[0]*anios[i]+d[1])

                return for_dt

            res = p.map(filtro_tendencia,i.tolist(),l.tolist(),j.tolist(),k.tolist())
            
            forecast_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            #forecast_dt = np.squeeze(np.stack(res))

            print(forecast_dt.shape)

            del(filtro_tendencia,res)

            p.close()

        else:

            def filtro_tendencia(i,l,j,k,anios=anios,forec=forecast): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)

                A = np.array([ anios, np.ones(anios.shape[0])])

                d = np.linalg.lstsq(A.T,y)[0]

                for_dt = forec[i,l,j,k]-(d[0]*anios[i]+d[1])

                return for_dt

            print("Paralelizando")

            res = p.map(filtro_tendencia,i.tolist(),l.tolist(),j.tolist(),k.tolist())
            
            print("Termino Paralelizacion")

            forecast_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            
            print(forecast_dt.shape)

            del(filtro_tendencia,res)

            p.close()
            
        return forecast_dt

    def ereg(self,forecast,observation,CV_opt):

        #calculo ensemble regression para cada modelo incluye la opcion de cal
        #cularlo con el factor de correccion de inflacion del ensamble K y si
        #quiero o no hacerlo en cross validacion

        [ntimes,nmembers,nlats,nlons] = forecast.shape

        #calculo primero si necesito corregir mi ensamble

        em = np.nanmean(forecast, axis = 1)

        Rmean = np.nanmean((observation-np.nanmean(observation, axis = 0 ))*(em-np.nanmean(em, axis = 0)), axis = 0)/(np.nanstd(em, axis = 0)*np.nanstd(observation, axis = 0))

        noise = np.nanmean(np.nanvar(forecast, axis = 1), axis = 0)

        Rbestp = Rmean * np.sqrt(1+(self.ensembles/(self.ensembles-1)*noise/np.nanvar(em, axis = 0)))

        epsbn = ntimes/(ntimes-1)*np.nanvar(observation, axis = 0)*(1-np.power(Rbestp,2))

        kmax = np.nanvar(em, axis = 0)/noise *(self.ensembles/(self.ensembles-1))*(1/np.power(Rmean,2)-1)

        #testeo

        K = kmax

        K[np.where(Rbestp<=1)] = 1

        K[np.where(K>1)] = 0

        K = np.repeat(np.repeat(K[np.newaxis,:,:], self.ensembles, axis=0)[np.newaxis,:,:,:],ntimes,axis=0)


        forecast_inf = np.repeat(np.mean(forecast, axis =1)[:,np.newaxis,:,:],nmembers,axis=1)*(1-K)+forecast*K
#       forecast_cr = np.empty_like(forecast)

        i = np.repeat(np.arange(ntimes,dtype = int),nmembers*nlats*nlons)

        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),nlats*nlons),ntimes)

        j = np.tile(np.repeat(np.arange(nlats,dtype = int),nlons),ntimes*nmembers)

        k = np.tile(np.arange(nlons,dtype = int),ntimes*nmembers*nlats)

        p = Pool(cores)

        p.clear()

        if CV_opt: #validacion cruzada ventana 1 anio

            print("Validacion cruzada")

            CV_matrix = np.logical_not(np.identity(ntimes))

            def ens_reg(i,l,j,k,CV_m = CV_matrix,obs = observation, forec=forecast): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)

                A = np.array([ y[CV_m[:,i]], np.ones((y.shape[0]-1))])

                d = np.linalg.lstsq(A.T,obs[CV_m[:,i],j,k])[0]

                for_cr = (d[0]*obs[i,j,k]+d[1])

                return for_cr

            res = p.map(ens_reg,i.tolist(),l.tolist(),j.tolist(),k.tolist())
            
            forecast_cr = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            
            print(forecast_cr.shape)

            del(ens_reg,res)

            p.close()

        else:

            def ens_reg(i,l,j,k, obs = observation, forec=forecast): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)

                A = np.array([ y, np.ones(y.shape[0])])

                d = np.linalg.lstsq(A.T,obs[:,j,k])[0]

                for_cr = (d[0]*obs[i,j,k]+d[1])

                return for_cr

            print("Paralelizando")

            res = p.map(ens_reg,i.tolist(),l.tolist(),j.tolist(),k.tolist())
            
            print("Termino Paralelizacion")

            forecast_cr = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            
            print(forecast_cr.shape)

            del(ens_reg,res)

            p.close()
                  
        #em = np.nanmean(forecast_inf, axis = 1)

        #Rmean = np.nanmean((observation-np.nanmean(observation, axis = 0 ))*(em-np.nanmean(em, axis = 0)), axis = 0)/(np.nanstd(em, axis = 0)*np.nanstd(observation, axis = 0))

        noise = np.nanmean(np.nanvar(forecast_inf, axis = 1), axis = 0)

        Rbestp = Rmean * np.sqrt(1+(self.ensembles/(self.ensembles-1)*noise/np.nanvar(em, axis = 0)))

        epsbn = ntimes/(ntimes-1)*np.nanvar(observation, axis = 0)*(1-np.power(Rbestp,2))

        #kmax = np.nanvar(em, axis = 0)/noise *(self.ensembles/(self.ensembles-1))*(1/np.power(Rmean,2)-1)

        return forecast_cr, Rmean, Rbestp, epsbn, kmax


    def pdf_eval(self,forecast,eps,observation):

        #obtengo la intensidad de la pdf calibrada en la observacion

        [ntimes,nmembers,nlat,nlon] = forecast.shape

        pdf_intensity = np.empty_like(forecast)

        i = np.repeat(np.arange(ntimes,dtype = int),nmembers*nlat*nlon)

        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),nlat*nlon),ntimes)

        j = np.tile(np.repeat(np.arange(nlat,dtype = int),nlon),ntimes*nmembers)

        k = np.tile(np.arange(nlon,dtype = int),ntimes*nmembers*nlat)
          
        p = Pool(cores)

        p.clear()

        def evaluo_pdf_normal(i,l,j,k,obs = observation,media = forecast,sigma = eps):

            pdf_intensity = stats.norm.pdf(obs[i,j,k], loc = media[i,l,j,k],scale = np.sqrt(sigma[j,k]))

            return pdf_intensity


        res = p.map (evaluo_pdf_normal,i.tolist(),l.tolist(), j.tolist(), k.tolist())

        p.close()
        
        pdf_intensity = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlat,nlon])

        del(p,res,evaluo_pdf_normal)

        #return res

        return pdf_intensity 
