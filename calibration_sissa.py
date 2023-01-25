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
#    parser.add_argument('--no-model', required=False, nargs='+', choices=\
#                        ['CFSv2', 'CanCM3', 'CanCM4', 'CM2p1', 'FLOR-A06',\
#                         'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'],
#                        dest='no_model', help="Models to be discarded")
    args = parser.parse_args()   # Extract dates from args
    file1 = open("configuracion", 'r')
    PATH = file1.readline().rstrip('\n')
    file1.close()
    lista = glob.glob(PATH + "modelos/*")
#    if args.no_model is not None:
#        lista = [i for i in lista if [line.rstrip('\n')
#                                      for line in open(i)][0] not in args.no_model]
    keys = ['nombre', 'instit', 'latn', 'lonn', 'miembros', 'plazos',\
            'fechai', 'fechaf', 'ext', 'rt_miembros']
    modelos = []
    for i in lista:
        lines = [line.rstrip('\n') for line in open(i)]
        modelos.append(dict(zip(keys, [lines[0], lines[1], lines[2], lines[3],\
                                       int(lines[4]), int(lines[5]), \
                                       int(lines[6]), int(lines[7]), \
                                       lines[8], int(lines[9])])))
    """ref dataset: depende de CI del prono y plazo.
    Ej: si IC prono es Jan y plazo 1 entonces FMA en primer tiempo 1982. Si IC
    prono es Dec y plazo 2 entonces FMA en primer tiempo es 1983."""
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    print("Calibrating " + args.variable[0] + " forecasts for " + SSS + " initialized in "
          + str(args.IC[0]) )

    print("Processing Observations")
    archivo2 = Path(PATH + 'DATA/Observations/' + 'obs_extremes_' + args.variable[0] + '_' +\
                    str(year_verif) + '_' + SSS + '_parameters.npz')
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
    quintiles = obs.computo_quintiles(obs_dt, args.CV) # Obtain tercile limits
    np.savez(archivo2, obs_dt=obs_dt, lats_obs=lats_obs, lons_obs=lons_obs,\
             quintiles=quintiles) #Save observed variables

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
