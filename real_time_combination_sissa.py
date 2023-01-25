"""
real time forecast using combination techniques developedin combination.py
"""
#!/usr/bin/env python
import argparse #parse command line options
import time #test time consummed
import datetime
import warnings
import glob
from pathlib import Path
import calendar
import numpy as np
import model
import ereg #apply ensemble regression to multi-model ensemble
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    """Defines parser data"""
    parser = argparse.ArgumentParser(description='Combining models')
    parser.add_argument('variable', type=str, nargs=1,\
            help='Variable to calibrate (prec or tref)')
    parser.add_argument('--IC', dest='IC', metavar='Date', type=str, nargs=1,
                        help='Initial conditions in "YYYY-MM-DD"')
    parser.add_argument('--leadtime', dest='leadtime', type=int, nargs=1,
                        help='Forecast leatime (in months, from 1 to 7)')
    subparsers = parser.add_subparsers(help="Combination technique")
    wpdf_parser = subparsers.add_parser('wpdf', help='weighted sum of calibrated PDFs')
    wsereg_parser = subparsers.add_parser('wsereg',
                                          help='Ereg with the weighted superensemble')
    count_parser = subparsers.add_parser('count', help='Count members in each bin')
    count_parser.set_defaults(ctech='count', wtech=['same'])
    count_parser.add_argument('--no-model', nargs='+', choices=['CFSv2','CanCM3','CanCM4',\
           'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'], dest='no_model',\
            help='Models to be discarded')
    wpdf_parser.set_defaults(ctech='wpdf')
    wpdf_parser.add_argument("--weight_tech", required=True, nargs=1,\
            choices=['pdf_int', 'mean_cor', 'same'], dest='wtech')
    wpdf_parser.add_argument('--no-model', nargs='+', choices=['CFSv2','CanCM3','CanCM4',\
           'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'],
                             dest='no_model', help='Models to be discarded')
    wsereg_parser.set_defaults(ctech='wsereg')
    wsereg_parser.add_argument("--weight_tech", required=True, nargs=1,\
            choices=['pdf_int', 'mean_cor', 'same'], dest='wtech',
                               help='Relative weight between models')
    wsereg_parser.add_argument('--no-model', nargs='+', choices=['CFSv2', 'CanCM3','CanCM4',\
            'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'],
                               dest='no_model', help='Models to be discarded')
    # Extract dates from args
    args = parser.parse_args()
    initialDate = datetime.datetime.strptime(args.IC[0], '%Y-%m-%d')
    iniy = initialDate.year
    inim = initialDate.month
    file1 = open("configuracion", 'r')
    PATH = file1.readline().rstrip('\n')
    file1.close()
    lista = glob.glob(PATH + "modelos/*")
    if args.no_model is not None: #si tengo que descartar modelos
        lista = [i for i in lista if [line.rstrip('\n') 
                                      for line in open(i)][0] not in args.no_model]
    keys = ['nombre', 'instit', 'latn', 'lonn', 'miembros', 'plazos',\
            'fechai', 'fechaf','ext', 'rt_miembros']
    modelos = []
    for i in lista:
        lines = [line.rstrip('\n') for line in open(i)]
        modelos.append(dict(zip(keys, [lines[0], lines[1], lines[2], lines[3],
                                       int(lines[4]), int(lines[5]),
                                       int(lines[6]), int(lines[7]),
                                       lines[8], int(lines[9])])))
    nmodels = len(modelos)
    ny = int(np.abs(coords['lat_n'] - coords['lat_s']) + 1)
    nx = int(np.abs (coords['lon_e'] - coords['lon_w']) + 1) #does for domains beyond greenwich
    nyears = int(modelos[0]['fechaf'] - modelos[0]['fechai'] + 1)
    if args.ctech == 'wpdf':
        print("not implemented - only wsereg")
        exit()
    elif args.ctech == 'wsereg':
        for_dt = np.array([]).reshape(nyears, ny, nx, 0) #[years lats lons members]
        prono_actual_dt = np.array([]).reshape(ny, nx, 0) #[lats lons members]
    if args.wtech[0] == 'pdf_int':
        print("not implemented - only mean_cor")
        exit()
    elif args.wtech[0] == 'mean_cor':
        rmean = np.array([]).reshape(ny, nx, 0)
    nmembers = []
    nmembersf = np.empty([nmodels], dtype=int)
    #defino ref dataset y target season
    seas = range(inim + args.leadtime[0], inim + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    print(args.variable[0] + ' IC:' + calendar.month_abbr[inim], 'Target season:' + SSS,
          args.ctech, args.wtech[0])
    #obtengo datos observados
    archivo = Path(PATH +  'DATA/Observations/obs_extremes_' + args.variable[0]+'_'+\
                   str(year_verif) + '_' + SSS + '_parameters.npz')
    data = np.load(archivo)
    quintiles = data['quintiles']
    j = 0
    for it in modelos:
        modelo = model.Model(it['nombre'], it['instit'], args.variable[0],\
                            it['latn'], it['lonn'], it['miembros'], \
                            it['plazos'], it['fechai'], it['fechaf'],\
                            it['ext'], it['rt_miembros'])
        [lats, lons, pronos] = modelo.select_real_time_months(inim, iniy,\
                                                             args.leadtime[0],
                                                             coords['lat_s'],
                                                             coords['lat_n'],
                                                             coords['lon_w'],
                                                             coords['lon_e'])
        """
        #aca deberia meter codigo que: -marque los pronosticos que son nan. cuente cuantos son nan
        #saque el modelo si todos los pronos son nan y pase al siguiente modelo. Reduzca el tamaño
        de los modelos
        """
        empty_forecast = np.sum(np.sum(np.isnan(pronos), axis=2), axis=1) == (nx * ny) # modificado
        print(empty_forecast)
        vacio = np.sum(empty_forecast) == pronos.shape[0]
        if not(vacio):
            #abro archivo modelo
            archivo = Path(PATH + 'DATA/calibrated_forecasts/'+ \
                           args.variable[0] + '_' + it['nombre'] + '_' +\
                           calendar.month_abbr[inim] + '_' + SSS +\
                           '_gp_01_hind_parameters.npz')
            data = np.load(archivo)
            a1 = data['a1']
            b1 = data['b1']
            #remove trend
            T = iniy - 1982
            f_dt = pronos - (b1 + T * a1)
            f_dt[empty_forecast, :, :] = np.nan # modificado
            #junto pronos actual
            prono_actual_dt = np.concatenate((prono_actual_dt,
                                              np.rollaxis(f_dt, 0, 3)), axis=2)
            nmembersf[j] = np.shape(f_dt)[0]
            nmembers.append(np.shape(f_dt)[0] - np.sum(empty_forecast)) # modificado
            #extraigo info del peso segun la opcion por la que elijo pesar
            Rm = data['Rm']
            rmean = np.concatenate((rmean, Rm[:, :, np.newaxis]), axis=2)
            Rm = []
        else:
            nmodels -= 1
        j = j + 1

    lat = lats
    lon = lons
    nlats = np.shape(lat)[0]
    nlons = np.shape(lon)[0]

  #calculo matriz de peso
    rmean[np.where(np.logical_or(rmean < 0, ~np.isfinite(rmean)))] = 0
    rmean[np.nansum(rmean[:, :, :], axis=2) == 0, :] = 1
    peso = rmean / np.tile(np.nansum(rmean, axis=2)[:, :, np.newaxis], [1, 1, nmodels])
    weight = np.tile(peso, (2, 1, 1, 1))  #2 nlat nlon nmodels

    weight = weight[0, :, :, :] / nmembers
    archivo = Path(PATH + 'DATA/combined_forecasts/' +\
                   args.variable[0]+'_mme_' + calendar.month_abbr[inim] +\
                   '_' + SSS + '_gp_01_' + args.wtech[0]+'_' + args.ctech +\
                   '_hind_parameters.npz')
    data = np.load(archivo)
    a_mme = data['a_mme']
    b_mme = data['b_mme']
    eps_mme = data['eps_mme']
    K_mme = data['K_mme']
    #peso prono actual
    # check nans
    empty_forecast = np.sum(np.sum(np.isnan(prono_actual_dt), axis=1), axis=0) == (nx * ny) # modificado
    prono_actual_dt = np.rollaxis(prono_actual_dt * np.repeat(weight, nmembersf, axis=2),
                                  2, 0)
    K_mme = K_mme[0, :, :, :]
    prono_actual_dt[empty_forecast, :, :] = np.nan
    prono_actual_dt = prono_actual_dt * K_mme + (1 - K_mme) *\
            np.nanmean(prono_actual_dt, axis = 0)
    #corrijo prono
    prono_cr = b_mme + a_mme * prono_actual_dt
    #obtains prob for each terciles,year and member
    prob_quint = ereg.probabilidad_quintiles(prono_cr, eps_mme, quintiles)
    prob_quint[:, empty_forecast, :, :] = np.nan
    #empty forecast to nana
    prob_quint_comb = np.nanmean(prob_quint, axis=1)

    #guardo los pronos
    archivo = Path(PATH + 'DATA/real_time_forecasts/' +\
                   args.variable[0]+'_extremes_mme_' + calendar.month_abbr[inim] +\
                   str(iniy) + '_' + SSS + '_gp_01_' + args.wtech[0]+'_' +\
                   args.ctech + '.npz')
    np.savez(archivo, prob_quint_comb=prob_quint_comb, lat=lat, lon=lon)
#=============================================================================
start = time.time()
#abro archivo donde guardo coordenadas
coordenadas = 'coords'
domain = [line.rstrip('\n') for line in open(coordenadas)]
coords = {'lat_s' : float(domain[1]),
        'lat_n' : float(domain[2]),
        'lon_w' : float(domain[3]),
        'lon_e' : float(domain[4])}
main()
end = time.time()
print(end - start)
# ============================================================================
