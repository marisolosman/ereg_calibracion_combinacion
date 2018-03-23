#modules
import numpy as np
import xarray as xr
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool
import warnings

cores = 9

#get netdf variables

def manipular_nc(archivo,variable,lat_name,lon_name,lats, latn, lonw, lone):

    dataset = xr.open_dataset(archivo,decode_times = False)

    var_out = dataset[variable].sel(**{lat_name: slice(lats,latn), lon_name: slice(lonw,lone)})

    lon = dataset[lon_name].sel(**{lon_name: slice(lonw,lone)})
    
    lat = dataset[lat_name].sel(**{lat_name: slice(lats,latn)})

    return var_out, lat, lon


class Model(object):
#caracteristicas comunes de todos los modelos
    def __init__(self, name, institution, var_name, lat_name, lon_name, miembros_ensamble, 
            leadtimes, hind_begin, hind_end,extension):

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

#selecciono los datos a partir del mes de IC y plazo, saco promedio trimestral
    
    def select_months(self, init_cond, target, lats, latn, lonw, lone):  
        #init_cond en meses y target en meses ()
       
        final_month = init_cond + 11

        if final_month > 12:
            flag_end = 1
            final_month = final_month - 12
        else:
            flag_end = 0
	
        ruta = '/datos/osman/nmme/monthly/'       

        #abro un archivo de ejempl
        file = ruta + 'prec_Amon_' + self.institution + '-' + self.name + '_'+'198201_r1i1p1_'+'198201-198212.'+ self.ext

        hindcast_length = self.hind_end - self.hind_begin + 1 

        [variable,latitudes,longitudes] = manipular_nc (file,self.var_name,self.lat_name,self.lon_name,lats,latn,lonw,lone)

        forecast = np.empty([hindcast_length,self.ensembles,latitudes.shape[0],longitudes.shape[0]])

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
                        forecast[ i-self.hind_begin, j-1,:,:] = np.nanmean(np.squeeze(np.array(variable))
                                [ target:target+3,:,:], axis = 0) #como todos tiene en el 0 el prono del propio

                    except RuntimeWarning:
                        forecast[ i-self.hind_begin, j-1,:,:] = np.NaN
                        print(i,j)                        
    
               #mes, target no hay que correrlo

                variable = []
	# Return values of interest: latitudes longitudes forecast
        return latitudes, longitudes, forecast

    #remuevo la tendencia del conjunto de datos: lo hago en crossvalidation con ventana de 1 anio

    def remove_trend(self, forecast, CV_opt):  #forecast 4-D array ntimes nensembles nlats nlons 
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

            def filtro_tendencia(i, l, j, k, anios = anios, forec = forecast): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)

                A = np.array([ anios, np.ones(anios.shape[0])])

                d = np.linalg.lstsq(A.T,y)[0]

                for_dt = forec[i,l,j,k]-(d[0]*anios[i]+d[1])

                return for_dt

            print("Paralelizando")

            res = p.map(filtro_tendencia, i.tolist(), l.tolist(), j.tolist(), k.tolist())
            
            print("Termino Paralelizacion")

            forecast_dt = np.reshape(np.squeeze(np.stack(res)),[ntimes, nmembers, nlats, nlons])
            
            print(forecast_dt.shape)

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

        noise = np.nanmean(np.nanvar(forecast, axis = 1), axis = 0) #media de la dispersion entre miembros

        #Rbest = Rm * sqrt( 1 +(m/(m-1) * N) /S )
        
        Rbest = Rm * np.sqrt(1+(self.ensembles/(self.ensembles-1) * noise)/signal)

        #epsbest = n/(n-1) * Varobs * (1-Rmean**2)

        epsbn =((ntimes-1)/(ntimes-2)) *  np.nanvar(observation, axis = 0)*(1-np.power(Rbest,2))

        #kmax**2 S/N * (m-1/m) * (1/R**2-1)
        kmax = signal/noise * (((self.ensembles-1)/self.ensembles) * (1/np.power(Rm,2)-1))

        # si kmax es amayor a 1 lo fuerzo a que sea 1

        kmax [np.where(kmax>1)] = 1

        #testeo

        K = np.zeros_like(epsbn)

        #si el error en la regresion positivo, entonces no cambio la dispersion del ensamble

        K[np.where(epsbn>=0)] = 1 * P

        #si el error en la regresion es negativo, voy a aplicar ereg solo con la media

        K[np.where(epsbn<0)] = 0
        

        K = np.repeat(np.repeat(K[np.newaxis,:,:], self.ensembles, axis=0)[np.newaxis,:,:,:],ntimes,axis=0)

        forecast_inf = np.repeat(np.nanmean(forecast, axis =1)[:,np.newaxis,:,:],
                nmembers,axis=1)*(1-K)+forecast*K

        #calulo otra vez los parametros que cambiaron

        noise = np.nanmean(np.nanvar(forecast_inf, axis = 1), axis = 0) #media de la dispersion entre miembros

        #Rbest = Rm * sqrt( 1 +(m/(m-1) * N) /S )
        
        Rbest = Rm * np.sqrt(1+(self.ensembles/(self.ensembles-1) * noise)/signal)

        #epsbest = n/(n-1) * Varobs * (1-Rmean**2)

        epsbn =((ntimes-1)/(ntimes-2)) *  np.nanvar(observation, axis = 0)*(1-np.power(Rbest,2))

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

                A = np.array([ y[CV_m[:,i]], np.ones((y.shape[0]-1))])

                d = np.linalg.lstsq(A.T,obs[CV_m[:,i],j,k])[0]

                for_cr = (d[0]*forec[i,l,j,k]+d[1])

                return for_cr

            res = p.map(ens_reg, i.tolist(), l.tolist(), j.tolist(), k.tolist())
            
            forecast_cr = np.reshape(np.squeeze(np.stack(res)),[ntimes,nmembers,nlats,nlons])

            del(ens_reg,res)

            p.close()

        else:

            def ens_reg(i, l, j, k, obs = observation, forec = forecast_inf): #forecast 4D

                y = np.nanmean(forec[:,:,j,k], axis = 1)

                A = np.array([ y, np.ones(y.shape[0])])

                d = np.linalg.lstsq(A.T,obs[:,j,k])[0]

                for_cr = (d[0]*forec[i,l,j,k]+d[1])

                return for_cr

            print("Paralelizando")

            res = p.map(ens_reg, i.tolist(), l.tolist(), j.tolist(), k.tolist())
            
            print("Termino Paralelizacion")

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

                    pdf_intensity = norm.pdf(obs[i,j,k], loc = media[i,l,j,k],scale = np.sqrt(sigma[j,k]))

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

        l = np.tile(np.repeat(np.arange(nmembers,dtype = int),
            nlats*nlons),ntimes)

        j = np.tile(np.repeat(np.arange(nlats,dtype = int),
            nlons),ntimes*nmembers)

        k = np.tile(np.arange(nlons,dtype = int),
                ntimes*nmembers*nlats)

        p = Pool(cores)

        p.clear()

        def evaluo_pdf_normal(i,l,j,k, terc = tercil,media = forecast,sigma = epsilon):

            pdf_cdf = norm.cdf(terc[:,i,j,k], loc = media[i,l,j,k],

                    scale = np.sqrt(sigma[j,k]))

            return pdf_cdf

        res = p.map (evaluo_pdf_normal, i.tolist(), l.tolist(),
                j.tolist(), k.tolist())

        p.close()

        prob_terciles = np.rollaxis(np.reshape(np.squeeze(np.stack(res)),
            [ntimes,nmembers,nlats,nlons,2]),4,0)

        return prob_terciles

