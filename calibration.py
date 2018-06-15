#!/usr/bin/env python

import argparse #para hacer el llamado desde consola
import time #tomar el tiempo que lleva el codigo
import numpy as np
import model #objeto y metodos asociados a los modelos
import observation # idem observaciones
import glob #listar archivos
import calendar #manejar meses del calendario
from pathlib import Path #manejar path

def main():
      
    parser = argparse.ArgumentParser(description = 'Calibrates model using Ensemble Regression.')
    parser.add_argument('variable', type = str, nargs = 1,\
            help ='Variable to calibrate (prec or temp)')
    parser.add_argument('IC', type = int, nargs = 1,\
            help = 'Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type = int, nargs = 1,\
            help = 'Forecast leatime (in months, from 1 to 7)')
    parser.add_argument('spread',  type = float, nargs = 1,\
            help = 'percentage of spread to retain (from 0 to 1)')
    parser.add_argument('--no-model', required = False, nargs = '+', choices = ['CFSv2', 'CanCM3', 'CanCM4', 'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'],
            dest = 'no_model', help = "Models to be discarded")  #Models to be excluded for calibration

    args = parser.parse_args()   # Extract dates from args
    p = args.spread[0]
    lista = glob.glob("/home/osman/proyectos/postdoc/modelos/*")
   
    if args.no_model is not None: 
        
        lista = [i for i in lista if [line.rstrip('\n') for line in open(i)][0] not in args.no_model]
        
    keys = ['nombre', 'instit', 'latn', 'lonn', 'miembros', 'plazos', 'fechai', 'fechaf','ext']
    modelos = []

    for i in lista:
       
        lines = [line.rstrip('\n') for line in open(i)]
        modelos.append(dict(zip(keys, [lines[0], lines[1], lines[2], lines[3], int(lines[4]), int(lines[5]), int(lines[6]), int(lines[7]), lines[8]])))
        
    """ref dataset (otro momento delirante): depende de CI del prono y plazo. 
    Ej: si IC prono es Jan y plazo 1 entonces FMA en primer tiempo 1982. Si IC prono es Dec y 
    plazo 2 entonces FMA en primer tiempo es 1983. Deberia charla con Wlad y Alfredo como resolver 
    esto mas eficientemente"""
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0]+3)
    sss = [i-12 if i>12 else i for i in seas ]
    year_verif = 1982 if seas[-1]<=12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    print("Processing Observations")
    archivo = Path('/datos/osman/nmme_output/obs_' + args.variable[0] + '_' + str(year_verif) + '_' + SSS + '.npz')

    if archivo.is_file():

        data = np.load(archivo)
        obs_dt = data['obs_dt']
        terciles = data['terciles']
        data.close()

    else:

        obs = observation.Observ('cpc', args.variable[0], 'Y', 'X', 1982, 2011)
        [lats_obs, lons_obs, obs_3m] = obs.select_months(calendar.month_abbr[sss[-1]], year_verif, coords['lat_s'], coords['lat_n'], coords['lon_w'], coords['lon_e']) #Select 3month observation
        obs_dt = obs.remove_trend(obs_3m, True) #Standardize and detrend observation
        terciles = obs.computo_terciles(obs_dt, True) # Obtain tercile limits
        categoria_obs = obs.computo_categoria(obs_dt, terciles)  #Define observed category
        np.savez(archivo, obs_dt = obs_dt,
                lats_obs = lats_obs, lons_obs = lons_obs, terciles = terciles, cat_obs =
                categoria_obs) #Save observed variables
    print("Processing Models")
    print (args.IC[0],SSS,'{:03}'.format(p))
    for it in modelos:

        output = Path('/datos/osman/nmme_output/cal_forecasts/'+args.variable[0] + '_' +\
                it['nombre'] + '_' + calendar.month_abbr[args.IC[0]] + '_' + SSS + '_gp_01_'\
                + 'p_' + '{:03}'.format(p) + '_hind.npz')
        print(output)

        if output.is_file():
            pass
        else:

            modelo = model.Model(it['nombre'], it['instit'], args.variable[0], it['latn'],\
                    it['lonn'], it['miembros'], it['plazos'], it['fechai'], it['fechaf'], it['ext'])
            print (modelo,args.IC[0],SSS,p)
            [lats, lons, pronos] = modelo.select_months(args.IC[0], args.leadtime[0], coords['lat_s'], coords['lat_n'], coords['lon_w'], coords['lon_e'])  #Select forecast
            pronos_dt = modelo.remove_trend(pronos, True)  #Standardize and detrend forecast
            for_terciles = modelo.computo_terciles(pronos_dt, True) #uncalibrated forecast terciles
            forecasted_category = modelo.computo_categoria(pronos_dt, for_terciles) #forecasted category
            """ Apply Ensemble Regression 
            Input: standardized forecast and observations
                    fraction of spread to retain
            Output: Corrected forecast
                    Observations vs Ensemble mean correlation
                    Observations vs Best ensemble correlation
                    Regression errors
                    Maximum correction factor admitted
                    Correction factor applied
            """
            [forecast_cr, Rmedio, Rmej, epsb, Kmax, K] = modelo.ereg(pronos_dt, obs_dt, p, True)
            prob_terc = modelo.probabilidad_terciles (forecast_cr, epsb, terciles)  #Compute the probability of occurrence of each tercile
            pdf_intensity = modelo.pdf_eval(forecast_cr, epsb, obs_dt)  #Compute pdf intensity at the observation point
            np.savez(output, lats = lats, lons = lons, pronos_dt = pronos_dt, pronos_cr = forecast_cr, 
                    eps = epsb, Rm = Rmedio, Rb = Rmej, K2 = Kmax, K = K, peso = pdf_intensity, 
                    prob_terc = prob_terc, forecasted_category = forecasted_category)  #Save model results

start = time.time()

if __name__=="__main__":
    coordenadas = 'coords'
    lines = [line.rstrip('\n') for line in open(coordenadas)]  #Get domain limits
    coords = {'lat_s': float(lines[1]), 
            'lat_n': float(lines [2]),
            'lon_w': float(lines[3]),
            'lon_e': float(lines[4])}
    main()
end = time.time()

print(end - start)
