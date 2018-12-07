#!/usr/bin/env python
"""open forecast and observations and calibrates models using ensemble regression"""
import argparse #para hacer el llamado desde consola
import time #tomar el tiempo que lleva el codigo
import glob #listar archivos
import calendar #manejar meses del calendario
from pathlib import Path #manejar path
import numpy as np

import model #objeto y metodos asociados a los modelos
import observation # idem observaciones

def main():
    parser = argparse.ArgumentParser(description='Calibrates model using Ensemble Regression.')
    parser.add_argument('variable', type=str, nargs=1,\
            help='Variable to calibrate (prec or tref)')
    parser.add_argument('IC', type=int, nargs=1,\
            help='Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type=int, nargs=1,\
            help='Forecast leatime (in months, from 1 to 7)')
    parser.add_argument('--CV', help='Croos-validated mode',
                        action= 'store_true')
    parser.add_argument('--no-model', required=False, nargs='+', choices=\
                        ['CFSv2', 'CanCM3', 'CanCM4', 'CM2p1', 'FLOR-A06',\
                         'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'],
                        dest='no_model', help="Models to be discarded")
    args = parser.parse_args()   # Extract dates from args
    lista = glob.glob("/home/osman/proyectos/postdoc/modelos/*")
    if args.no_model is not None:
        lista = [i for i in lista if [line.rstrip('\n')
                                      for line in open(i)][0] not in args.no_model]

    keys = ['nombre', 'instit', 'latn', 'lonn', 'miembros', 'plazos',\
            'fechai', 'fechaf', 'ext', 'rt_miembros']
    modelos = []
    for i in lista:
        lines = [line.rstrip('\n') for line in open(i)]
        modelos.append(dict(zip(keys, [lines[0], lines[1], lines[2], lines[3],\
                                       int(lines[4]), int(lines[5]), \
                                       int(lines[6]), int(lines[7]), \
                                       lines[8], int(lines[9])])))
    """ref dataset (otro momento delirante): depende de CI del prono y plazo.
    Ej: si IC prono es Jan y plazo 1 entonces FMA en primer tiempo 1982. Si IC
    prono es Dec y plazo 2 entonces FMA en primer tiempo es 1983. Deberia
    charla con Wlad y Alfredo como resolver esto mas eficientemente"""
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    print("Processing Observations")
    archivo = Path('/datos/osman/nmme_output/obs_' + args.variable[0] + '_' +\
                       str(year_verif) + '_' + SSS + '.npz')
    if archivo.is_file():
        if args.CV:
            data = np.load(archivo)
            obs_dt = data['obs_dt']
            terciles = data['terciles']
            data.close()
    else:
        if args.CV:
            if args.variable[0] == 'prec':
                obs = observation.Observ('cpc', args.variable[0], 'Y', 'X', 1982,
                                         2011)
            else:
                obs = observation.Observ('ghcn_cams', args.variable[0], 'Y', 'X', 1982,
                                         2011)


            [lats_obs, lons_obs, obs_3m] = obs.select_months(calendar.month_abbr[\
                                                                                 sss[-1]],
                                                             year_verif,
                                                             coords['lat_s'],
                                                             coords['lat_n'],
                                                             coords['lon_w'],
                                                             coords['lon_e'])
            obs_dt = obs.remove_trend(obs_3m, args.CV) #Standardize and detrend observation
            terciles = obs.computo_terciles(obs_dt, args.CV) # Obtain tercile limits
            categoria_obs = obs.computo_categoria(obs_dt, terciles)  #Define observed category
            np.savez(archivo, obs_dt=obs_dt, lats_obs=lats_obs, lons_obs=lons_obs,\
                     terciles=terciles, cat_obs=categoria_obs)
    if np.logical_not(args.CV):
        archivo2 = Path('/datos/osman/nmme_output/obs_' + args.variable[0] + '_' +\
                       str(year_verif) + '_' + SSS + '_parameters.npz')
        if archivo2.is_file():
            data = np.load(archivo2)
            obs_dt = data['obs_dt']
            terciles =data['terciles']
            data.close()
        else:
            if args.variable[0] == 'prec':
                obs = observation.Observ('cpc', args.variable[0], 'Y', 'X', 1982,
                                         2011)
            else:
                obs = observation.Observ('ghcn_cams', args.variable[0], 'Y', 'X', 1982,
                                         2011)
            [lats_obs, lons_obs, obs_3m] = obs.select_months(calendar.month_abbr[\
                                                                                 sss[-1]],
                                                                      year_verif,
                                                                      coords['lat_s'],
                                                                      coords['lat_n'],
                                                                      coords['lon_w'],
                                                                      coords['lon_e'])
            obs_dt = obs.remove_trend(obs_3m, args.CV) #Standardize and detrend observation
            terciles = obs.computo_terciles(obs_dt, args.CV) # Obtain tercile limits
            np.savez(archivo2, obs_dt=obs_dt, lats_obs=lats_obs, lons_obs=lons_obs,\
                     terciles=terciles) #Save observed variables

    print("Processing Models")
    RUTA = '/datos/osman/nmme_output/cal_forecasts/'
    for it in modelos:
        output = Path(RUTA, args.variable[0] + '_' + it['nombre'] + '_' + \
                      calendar.month_abbr[args.IC[0]] + '_' + SSS + \
                      '_gp_01_hind.npz')
        if output.is_file():
            if args.CV:
                pass
            else:
                data = np.load(output)
                pdf_intensity = data['peso']
                data.close()
        else:
            modelo = model.Model(it['nombre'], it['instit'], args.variable[0],\
                                 it['latn'], it['lonn'], it['miembros'], \
                                 it['plazos'], it['fechai'], it['fechaf'],\
                                 it['ext'], it['rt_miembros'])
            print(modelo, args.IC[0], SSS)
            [lats, lons, pronos] = modelo.select_months(args.IC[0], \
                                                        args.leadtime[0], \
                                                        coords['lat_s'], \
                                                        coords['lat_n'],
                                                        coords['lon_w'],
                                                        coords['lon_e'])
            pronos_dt = modelo.remove_trend(pronos, True)
            for_terciles = modelo.computo_terciles(pronos_dt, True)
            forecasted_category = modelo.computo_categoria(pronos_dt, for_terciles)
            [forecast_cr, Rmedio, Rmej, epsb, K] = modelo.ereg(pronos_dt,\
                                                               obs_dt,
                                                               True)
            pdf_intensity = modelo.pdf_eval(forecast_cr, epsb, obs_dt)
            if args.CV:
                prob_terc = modelo.probabilidad_terciles(forecast_cr, epsb,\
                                                         terciles)
                np.savez(output, lats=lats, lons=lons, pronos_dt=pronos_dt,
                         pronos_cr=forecast_cr, eps=epsb, Rm=Rmedio, Rb=Rmej, K=K,
                         peso=pdf_intensity, prob_terc=prob_terc,
                         forecasted_category=forecasted_category)
        if np.logical_not(args.CV):
            output2 = Path(RUTA, args.variable[0] + '_' + it['nombre'] + '_' + \
                          calendar.month_abbr[args.IC[0]] + '_' + SSS + \
                          '_gp_01_hind_parameters.npz')
            if output2.is_file():
                pass
            else:
                modelo = model.Model(it['nombre'], it['instit'], args.variable[0],\
                                 it['latn'], it['lonn'], it['miembros'], \
                                 it['plazos'], it['fechai'], it['fechaf'],\
                                 it['ext'], it['rt_miembros'])
                print(modelo, args.IC[0], SSS)
                [lats, lons, pronos] = modelo.select_months(args.IC[0], \
                                                            args.leadtime[0], \
                                                            coords['lat_s'], \
                                                            coords['lat_n'],
                                                            coords['lon_w'],
                                                            coords['lon_e'])
                [pronos_dt, a1, b1] = modelo.remove_trend(pronos, args.CV)
                [a2, b2, Rmedio, Rmej, epsb, K] = modelo.ereg(pronos_dt,
                                                              obs_dt,
                                                              args.CV)
                np.savez(output2, lats=lats, lons=lons, pronos_dt=pronos_dt,
                         a1=a1, b1=b1, a2=a2, b2=b2, eps=epsb, Rm=Rmedio, Rb=Rmej, K=K,
                         peso=pdf_intensity)
#================================================================================================
start = time.time()
if __name__=="__main__":
    coordenadas = 'coords'
    domain = [line.rstrip('\n') for line in open(coordenadas)]  #Get domain limits
    coords = {'lat_s': float(domain[1]),
              'lat_n': float(domain[2]),
              'lon_w': float(domain[3]),
              'lon_e': float(domain[4])}
    main()
end = time.time()
print(end - start)
