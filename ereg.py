"""functions to combine ensemble forecast using ereg"""
import numpy as np
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool

CORES = 9

def ensemble_regression(forecast, observation, CV_opt):

    [ntimes, nmembers, nlats, nlons] = forecast.shape
    p = Pool(CORES)
    p.clear()
    if CV_opt: #validacion cruzada ventana 1 anio
        i = np.arange(ntimes, dtype=int)
        CV_matrix = np.logical_not(np.identity(ntimes))
        def compute_clim(i, CV_m=CV_matrix, obs=observation, forec=forecast):
            #computes climatologies under CV
            obs_c = np.mean(obs[CV_m[:, i], :, :], axis=0)
            em_c = np.mean(np.mean(forec[CV_m[:, i], :, :, :], axis=1),
                           axis=0)
            return obs_c, em_c
        res = p.map(compute_clim, i.tolist())
        res = np.stack(res, axis=1)
        obs_c = res[0, :, :, :]
        em_c = res[1, :, :, :]
    else:
        obs_c = np.mean(observation, axis=0)
        em_c = np.mean(np.mean(forecast, axis=1), axis=0)

    em = np.mean(forecast, axis=1)
    obs_var = np.sum(np.power(observation - obs_c, 2), axis=0) / ntimes
    signal = np.sum(np.power(em - em_c, 2), axis=0) / ntimes
    Rm = np.nanmean((observation - obs_c) * (em - em_c), axis=0) / np.sqrt(
        obs_var * signal)
    noise = np.nanmean(np.nanvar(forecast, axis=1), axis=0) #noise
    #Rbest = Rm sqrt( 1 + (m/(m - 1) * N) /S )
    Rbest = Rm * np.sqrt(1 + (nmembers / (nmembers - 1) * noise) / signal)
    #epsbest = n/(n-1) * Varobs * (1-Rmean**2)
    epsbn = (ntimes / (ntimes - 1)) *  obs_var * (1 - np.power(Rbest, 2))
    #kmax**2 S/N * (m-1/m) * (1/R**2-1)
    kmax = signal / noise * (((nmembers - 1)/nmembers) *
                             (1 / np.power(Rm, 2) - 1))
    # si kmax es amayor a 1 lo fuerzo a que sea 1
    kmax[kmax > 1] = 1
    #testeo
    K = np.zeros_like(epsbn)
    #if epsbn is positive spread remains the same
    K[epsbn >= 0] = 1
    #if epsbn is negative spread changes
    K[epsbn < 0] = kmax[epsbn < 0]
    K = np.repeat(np.repeat(K[np.newaxis, :, :], nmembers,
                            axis=0)[np.newaxis, :, :, :], ntimes,
                  axis=0)
    forecast_inf = forecast * K + (1 - K) *\
            np.rollaxis(np.repeat(em[np.newaxis, :, :, :], nmembers,
                                  axis=0), 1)
    #compute Rbest and epsbn again
    noise = np.nanmean(np.nanvar(forecast_inf, axis=1), axis=0) #noise
    #Rbest = Rm sqrt( 1 + (m/(m - 1) * N) /S )
    Rbest = Rm * np.sqrt(1 + (nmembers / (nmembers - 1) * noise) / signal)
    #epsbest = n/(n-1) * Varobs * (1-Rmean**2)
    epsbn = (ntimes / (ntimes - 1)) *  obs_var * (1 - np.power(Rbest, 2))
    print(np.sum(np.sum(np.sum(epsbn<0))))

    #ahora calculo la regresion
    i = np.repeat(np.arange(ntimes, dtype=int), nmembers * nlats * nlons)
    l = np.tile(np.repeat(np.arange(nmembers, dtype=int), nlats* nlons), ntimes)
    j = np.tile(np.repeat(np.arange(nlats, dtype=int), nlons), ntimes * nmembers)
    k = np.tile(np.arange(nlons, dtype=int), ntimes * nmembers * nlats)
    p = Pool(CORES)
    p.clear()

    if CV_opt: #validacion cruzada ventana 1 anio
        print("Validacion cruzada")
        CV_matrix = np.logical_not(np.identity(ntimes))
        def ens_reg(i, l, j, k, CV_m=CV_matrix, obs=observation, forec=forecast_inf):
            y = np.nanmean(forec[:, :, j, k], axis=1)
            A = np.vstack([y[CV_m[:, i]], np.ones((y.shape[0] - 1))])
            m, c = np.linalg.lstsq(A.T, obs[CV_m[:, i], j, k])[0]
            for_cr = m * forec[i, l, j, k] + c
            return for_cr

        res = p.map(ens_reg, i.tolist(), l.tolist(), j.tolist(), k.tolist())
        forecast_cr = np.reshape(np.squeeze(np.stack(res)), [ntimes, nmembers, nlats, nlons])
        del(ens_reg, res)
        p.close()
        return forecast_cr, Rm, Rbest, epsbn, kmax, K
    else:
        j = np.repeat(np.arange(nlats, dtype=int), nlons)
        k = np.tile(np.arange(nlons, dtype=int), nlats)
        def ens_reg(j, k, obs=observation, forec=forecast_inf): #forecast 4D
            y = np.nanmean(forec[:, :, j, k], axis=1)
            A = np.vstack([y, np.ones(y.shape[0])])
            m, c = np.linalg.lstsq(A.T, obs[:, j, k])[0]
            #for_cr = m * forec[i, l, j, k] + c
            return m, c
        res = p.map(ens_reg, j.tolist(), k.tolist())
        res = np.stack(res, axis=1)
        a2 = np.reshape(res[0, :], [nlats, nlons])
        b2 = np.reshape(res[1, :], [nlats, nlons])
        del(ens_reg, res)
        p.close()
        return a2, b2, Rm, Rbest, epsbn, kmax, K

def probabilidad_terciles(forecast, epsilon, tercil):
    """computes cpdf until tercil limits"""
    if np.ndim(forecast) == 4:
        [ntimes, nmembers, nlats, nlons] = forecast.shape
        i = np.repeat(np.arange(ntimes, dtype=int), nmembers * nlats * nlons)
        l = np.tile(np.repeat(np.arange(nmembers, dtype=int), nlats * nlons),
                    ntimes)
        j = np.tile(np.repeat(np.arange(nlats, dtype=int), nlons),
                    ntimes * nmembers)
        k = np.tile(np.arange(nlons, dtype=int), ntimes * nmembers * nlats)
        p = Pool (CORES)
        p.clear()
        def evaluo_pdf_normal(i, l, j, k, terc=tercil, media=forecast, sigma=epsilon):
            pdf_cdf = norm.cdf(terc[:, i, j, k], loc=media[i, l, j, k],
                               scale=np.sqrt(sigma[j, k]))
            return pdf_cdf

        res = p.map(evaluo_pdf_normal, i.tolist(), l.tolist(), j.tolist(),
                    k.tolist())
        p.close()
        prob_terciles = np.rollaxis(np.reshape(np.squeeze(np.stack(res)),
                                               [ntimes, nmembers, nlats, nlons,
                                                2]), 4, 0)
    else:
        [nmembers, nlats, nlons] = forecast.shape
        l = np.repeat(np.arange(nmembers, dtype=int), nlats * nlons)
        j = np.tile(np.repeat(np.arange(nlats, dtype=int), nlons), nmembers)
        k = np.tile(np.arange(nlons, dtype=int), nmembers * nlats)
        p = Pool (CORES)
        p.clear()
        def evaluo_pdf_normal(l, j, k, terc=tercil, media=forecast, sigma=epsilon):
            pdf_cdf = norm.cdf(terc[:, j, k], loc=media[l, j, k],
                               scale=np.sqrt(sigma[j, k]))
            return pdf_cdf

        res = p.map(evaluo_pdf_normal, l.tolist(), j.tolist(), k.tolist())
        p.close()
        prob_terciles = np.rollaxis(np.reshape(np.squeeze(np.stack(res)),
                                               [nmembers, nlats, nlons, 2]),
                                    3, 0)

    return prob_terciles
