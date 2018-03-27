#modules
import numpy as np
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool

cores = 9

def ensemble_regression(forecast, observation, P, CV_opt):

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

    noise = np.nanmean(np.nanvar(forecast, axis = 1), axis = 0) 
    #media de la dispersion entre miembros

    #Rbest = Rm * sqrt( 1 +(m/(m-1) * N) /S )

    Rbest = Rm * np.sqrt(1+(nmembers/(nmembers-1) * noise)/signal)

    #epsbest = n/(n-1) * Varobs * (1-Rmean**2)

    epsbn =((ntimes-1)/(ntimes-2)) *  np.nanvar(observation, 
            axis = 0)*(1-np.power(Rbest,2))

    #kmax**2 S/N * (m-1/m) * (1/R**2-1)

    kmax = signal/noise * (((nmembers-1)/nmembers) 
            * (1/np.power(Rm,2)-1))

    # si kmax es amayor a 1 lo fuerzo a que sea 1

    kmax [np.where(kmax>1)] = 1

    #testeo

    K = np.zeros_like(epsbn)

    #si el error en la regresion positivo, entonces no cambio la dispersion 
    #del ensamble

    K[np.where(epsbn>=0)] = 1 * P

    #si el error en la regresion es negativo, voy a aplicar ereg solo con la media

    K[np.where(epsbn<0)] = 0

    K = np.repeat(np.repeat(K[np.newaxis,:,:], 
        nmembers, axis=0)[np.newaxis,:,:,:],ntimes,axis=0)

    forecast_inf = np.repeat(np.nanmean(forecast, axis =1)[:,np.newaxis,:,:],

            nmembers,axis=1)*(1-K)+forecast*K

    #calulo otra vez los parametros que cambiaron

    noise = np.nanmean(np.nanvar(forecast_inf, axis = 1), axis = 0) 
    #media de la dispersion entre miembros

    #Rbest = Rm * sqrt( 1 +(m/(m-1) * N) /S )
    
    Rbest = Rm * np.sqrt(1+(nmembers/(nmembers-1) * noise)/signal)

                                                                                            #epsbest = n/(n-1) * Varobs * (1-Rmean**2)

    epsbn =((ntimes-1)/(ntimes-2)) *  np.nanvar(observation, 
            axis = 0)*(1-np.power(Rbest,2))

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

def probabilidad_terciles (forecast,epsilon,tercil):

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
    
    p = Pool (cores)

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

