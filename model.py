#modules
import numpy as np
import xarray as xr
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool
import warnings

cores = 9

def manipular_nc(archivo,variable,lat_name,lon_name,lats, latn, lonw, lone):
    #get netdf variables
    dataset = xr.open_dataset(archivo,decode_times = False)
    var_out = dataset[variable].sel(**{lat_name: slice(lats,latn), lon_name: slice(lonw,lone)})
    lon = dataset[lon_name].sel(**{lon_name: slice(lonw,lone)})
    lat = dataset[lat_name].sel(**{lat_name: slice(lats,latn)})
    return var_out, lat, lon

class Model(object):
    def __init__(self, name, institution, var_name, lat_name, lon_name, miembros_ensamble, 
            leadtimes, hind_begin, hind_end,extension):
        #caracteristicas comunes de todos los modelos
        self.name = name
        self.institution = institution
        self.ensembles = miembros_ensamble
        self.leadtimes = leadtimes
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.hind_begin = hind_begin
        self.hind_end = hind_end
        self.ext = extension
        pass
    #imprimir caracteristicas generales del modelo
    def __str__(self):
        return "%s is a model from %s and has %s ensemble members and %s leadtimes" % (self.name, 
                self.institution, self.ensembles, self.leadtimes)

#comienzo la definición de los métodos
    
    def select_months(self, init_cond, target, lats, latn, lonw, lone): 
        #selecciono los datos a partir del mes de IC y plazo, saco promedio trimestral
        #init_cond en meses y target en meses ()
       
        final_month = init_cond + 11

        if final_month > 12:
            flag_end = 1
            final_month = final_month - 12
        else:
            flag_end = 0
	
        ruta = '/datos/osman/nmme/monthly/'       
        #abro un archivo de ejemplo
        hindcast_length = self.hind_end - self.hind_begin + 1 
        forecast = np.empty([hindcast_length,self.ensembles,int(np.abs(latn-lats))+1,
            int(np.abs(lonw-lone))+1])
        #loop sobre los anios del hindcast period
        for i in np.arange(self.hind_begin,self.hind_end+1):
            
            for j in np.arange(1,self.ensembles+1):
                
                file = ruta + 'prec_Amon_' + self.institution + '-' + self.name + '_' + str(
                        i) + '{:02d}'.format(init_cond)+'_r'+str(j)+'i1p1_'+str(i
                                )+'{:02d}'.format(init_cond) + '-' + str(i+
                                        flag_end)+'{:02d}'.format(final_month)+'.'+self.ext

                [variable,latitudes,longitudes] = manipular_nc (file,self.var_name,self.lat_name,
                        self.lon_name,lats,latn,lonw,lone)

                with warnings.catch_warnings():

                    warnings.filterwarnings('error')

                    try:
                        forecast[ i-self.hind_begin, j-1,:,:] = np.nanmean(np.squeeze(
                            np.array(variable))[ target:target+3,:,:], axis = 0) 
                        #como todos tiene en el 0 el prono del propio
     
                    except RuntimeWarning:
                        forecast[ i-self.hind_begin, j-1,:,:] = np.NaN
                        print(i,j)                        
    
                variable = []

	# Return values of interest: latitudes longitudes forecast
        return latitudes, longitudes, forecast

    def remove_trend(self, forecast, CV_opt):
        #remuevo la tendencia del conjunto de datos: lo hago en crossvalidation con ventana de 1 anio
        #forecast 4-D array ntimes nensembles nlats nlons 
        #CV_opt boolean

        [ntimes,nmembers,nlats,nlons] = forecast.shape
        anios = np.arange(ntimes)
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

                y = np.nanmean(forec[:,:,j,k], axis = 1) #media del ensamble
                A = np.vstack([ anios[CV_m[:,i]], np.ones((anios.shape[0]-1))])
                m, c = np.linalg.lstsq(A.T,y[CV_m[:,i]])[0]
                for_dt = forec[i,l,j,k]-(m*anios[i]+c)
                return for_dt

            res = p.map(filtro_tendencia,i.tolist(),l.tolist(),j.tolist(),k.tolist())
            forecast_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            del(filtro_tendencia,res)
            p.close()

        else:

            def filtro_tendencia(i, l, j, k, anios = anios, forec = forecast): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)
                A = np.vstack([ anios, np.ones(anios.shape[0])])
                m, c = np.linalg.lstsq(A.T,y)[0]
                for_dt = forec[i,l,j,k]-(m*anios[i]+c)
                return for_dt

            print("Paralelizando")
            res = p.map(filtro_tendencia, i.tolist(), l.tolist(), j.tolist(), k.tolist())
            print("Termino Paralelizacion")
            forecast_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes, nmembers, nlats, nlons])
            del(filtro_tendencia,res)
            p.close()
            
        return forecast_dt

    def ereg(self, forecast, observation, P, CV_opt):
        #calculo ensemble regression para cada modelo incluye la opcion de cal
        #cularlo con el factor de correccion de inflacion del ensamble K y si
        #quiero o no hacerlo en cross validacion

        [ntimes,nmembers,nlats,nlons] = forecast.shape
        #calculo primero si necesito corregir mi ensamble
        em = np.nanmean(forecast, axis = 1) #media del ensamble
        Rm = np.nanmean((observation - np.nanmean(observation, axis = 0 )) * 
                (em-np.nanmean(em, axis = 0)), axis = 0) / (np.nanstd(em, axis = 0) * 
                        np.nanstd(observation, axis = 0)) #correlacion entre ens mean y obs

        signal = np.nanvar (em, axis = 0) #senial: varianza de la media del ensamble
        noise = np.nanmean(np.nanvar(forecast, axis = 1), axis = 0) #media del spread
        #Rbest = Rm * sqrt( 1 +(m/(m-1) * N) /S )
        Rbest = Rm * np.sqrt(1+(self.ensembles/(self.ensembles-1) * noise)/signal)
        #epsbest = n/(n-1) * Varobs * (1-Rmean**2)
        epsbn =((ntimes)/(ntimes-1)) *  np.nanvar(observation, axis = 0)*(1-np.power(Rbest,2))
        #kmax**2 S/N * (m-1/m) * (1/R**2-1)
        kmax = signal/noise * (((self.ensembles-1)/self.ensembles) * (1/np.power(Rm,2)-1))
        # si kmax es amayor a 1 lo fuerzo a que sea 1
        kmax [np.where(kmax>1)] = 1
        #testeo
        K = np.zeros_like(epsbn)
        #si el error en la regresion positivo, entonces cambio la dispersion del ensamble segun p
        K[np.where(epsbn>=0)] = 1 * P
        #si el error en la regresion es negativo, voy a aplicar ereg solo con la media
        K[np.where(epsbn<0)] = 0
        K = np.repeat(np.repeat(K[np.newaxis,:,:], self.ensembles, axis=0)
                [np.newaxis,:,:,:],ntimes,axis=0)
        forecast_inf = np.repeat(np.nanmean(forecast, axis =1)[:,np.newaxis,:,:],
                nmembers,axis=1)*(1-K)+forecast*K
        #calulo otra vez los parametros que cambiaron
        noise = np.nanmean(np.nanvar(forecast_inf, axis = 1), axis = 0) #media del spread
        #Rbest = Rm * sqrt( 1 +(m/(m-1) * N) /S )
        Rbest = Rm * np.sqrt(1+(self.ensembles/(self.ensembles-1) * noise)/signal)
        #epsbest = n/(n-1) * Varobs * (1-Rmean**2)
        epsbn =((ntimes)/(ntimes)) *  np.nanvar(observation, axis = 0)*(1-np.power(Rbest,2))

        #ahora calculo la regresion

        i = np.repeat(np.arange(ntimes,dtype = int),nmembers*nlats*nlons)
        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),nlats*nlons),ntimes)
        j = np.tile(np.repeat(np.arange(nlats,dtype = int),nlons),ntimes*nmembers)
        k = np.tile(np.arange(nlons,dtype = int),ntimes*nmembers*nlats)
        p = Pool(cores)
        p.clear()

        if CV_opt: #validacion cruzada ventana 1 anio

            print("Validacion cruzada")
            CV_matrix = np.logical_not(np.identity(ntimes))

            def ens_reg(i,l,j,k,CV_m = CV_matrix,obs = observation, forec=forecast_inf): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)
                A = np.vstack([ y[CV_m[:,i]], np.ones((y.shape[0]-1))])
                m, c = np.linalg.lstsq(A.T, obs[CV_m[:,i],j,k])[0]
                for_cr = m * forec[i,l,j,k] + c 
                return for_cr

            res = p.map(ens_reg, i.tolist(), l.tolist(), j.tolist(), k.tolist())
            forecast_cr = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            del(ens_reg,res)
            p.close()

        else:

            def ens_reg(i, l, j, k, obs = observation, forec = forecast_inf): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)
                A = np.vstack([ y, np.ones(y.shape[0])])
                m, c = np.linalg.lstsq(A.T, obs[:,j,k])[0]
                for_cr = m * forec[i,l,j,k] + c
                return for_cr

            res = p.map(ens_reg, i.tolist(), l.tolist(), j.tolist(), k.tolist())
            forecast_cr = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])
            print(forecast_cr.shape)
            del(ens_reg,res)
            p.close()
                  
        return forecast_cr, Rm, Rbest, epsbn, kmax, K

    def pdf_eval(self,forecast,eps,observation):
        #obtengo la intensidad de la pdf calibrada en la observacion

        [ntimes,nmembers,nlat,nlon] = forecast.shape

        i = np.repeat(np.arange(ntimes,dtype = int),nmembers*nlat*nlon)
        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),nlat*nlon),ntimes)
        j = np.tile(np.repeat(np.arange(nlat,dtype = int),nlon),ntimes*nmembers)
        k = np.tile(np.arange(nlon,dtype = int),ntimes*nmembers*nlat)
        p = Pool(cores)
        p.clear()

        def evaluo_pdf_normal(i,l,j,k,obs = observation,media = forecast,sigma = eps):

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    pdf_intensity = norm.pdf(obs[i,j,k], loc = media[i,l,j,k], 
                            scale = np.sqrt(sigma[j,k]))
                except RuntimeWarning:
                    pdf_intensity = np.NaN
                    print(sigma[j,k],j,k)

            return pdf_intensity


        res = p.map (evaluo_pdf_normal, i.tolist(), l.tolist(), j.tolist(), k.tolist())
        p.close()
        pdf_intensity = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlat,nlon])
        del(p,res,evaluo_pdf_normal)
        #return res
        return pdf_intensity

    def probabilidad_terciles (self,forecast,epsilon,tercil):
        #calcula la probabilidad acumulada hasta los limites de cada uno de los terciles
        #para cada anio y cada miembro del ensamble. Asume PDF normal con unger et al 2008

        [ntimes, nmembers, nlats, nlons] = forecast.shape

        i = np.repeat(np.arange(ntimes,dtype = int),nmembers*nlats*nlons)
        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),nlats*nlons),ntimes)
        j = np.tile(np.repeat(np.arange(nlats,dtype = int),nlons),ntimes*nmembers)
        k = np.tile(np.arange(nlons,dtype = int),ntimes*nmembers*nlats)
        p = Pool(cores)
        p.clear()

        def evaluo_pdf_normal(i,l,j,k, terc = tercil,media = forecast,sigma = epsilon):

            pdf_cdf = norm.cdf(terc[:,i,j,k], loc = media[i,l,j,k],scale = np.sqrt(sigma[j,k]))
            return pdf_cdf

        res = p.map (evaluo_pdf_normal, i.tolist(), l.tolist(),j.tolist(), k.tolist())
        p.close()
        prob_terciles = np.rollaxis(np.reshape(np.squeeze(np.stack(res)),
            [ntimes,nmembers,nlats,nlons,2]),4,0)

        return prob_terciles

    def computo_terciles (self, forecast, CV_opt):
        #calculo los limites de los terciles para el modelo

        [ntimes, nmembers, nlats, nlons] = forecast.shape

        if CV_opt: #validacion cruzada ventana 1 anio
            
            i = np.arange(ntimes)
            p = Pool(cores)
            p.clear()
            print("Validacion cruzada")
            CV_matrix = np.logical_not(np.identity(ntimes))

            def cal_terciles(i, CV_m = CV_matrix, forec = forecast):
                A = np.sort(np.rollaxis(np.reshape(forecast[CV_m[:,i],:,:,:], 
                    [(ntimes-1)*nmembers, nlats, nlons]), 0, 3), axis = -1, kind = 'quicksort')
                upper = A[:,:, np.int(np.round((ntimes-1)*nmembers/3)-1)]
                lower = A[:,:, np.int(np.round((ntimes-1)*nmembers/3*2)-1)]

                return upper, lower

            res = p.map(cal_terciles, i.tolist())
            terciles = np.stack(res, axis = 1)

            del(cal_terciles, res)
            p.close()

        else:
            A = np.sort(np.rollaxis(reshape(forecast, [ntimes*nmebers, nlat, nlons]),
                0, 3), axis = -1, kind = 'quicksort')
            upper = A[:,:, np.int(np.round(ntimes*nmembers/3)-1)]
            lower = A[:,:, np.int(np.round(ntimes*nmembers/3*2)-1)]
            terciles = np.rollaxis(np.stack([upper,lower], axis = 2), 2, 0)

        return terciles

    def computo_categoria (self, forecast, tercil):
        #clasifico cada anio y miembro del ensamble segun el tercil donde cayo: Below, Normal, Above

        [ntimes, nmembers, nlats, nlons] = forecast.shape
        #calculo el tercil pronosticado
        forecast_terciles = np.empty([3, ntimes, nmembers, nlats, nlons])
        for i in np.arange(ntimes):
            #above normal
            forecast_terciles[0,i,:,:,:] = forecast[i,:,:,:] <= tercil[0,i,:,:]
            #below normal
            forecast_terciles[2,i,:,:,:] = forecast[i,:,:,:] >= tercil[1,i,:,:]

        #near normal
        forecast_terciles[1,:,:,:,:] = np.logical_not(np.logical_or(forecast_terciles[0,:,:,:,:],
            forecast_terciles[2,:,:,:,:]))

        return forecast_terciles

