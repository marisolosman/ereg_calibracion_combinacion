"""
this code plots model weights for each IC and leadtime
"""
#!/usr/bin/env python

import argparse #parse command line options
import time #test time consummed
import numpy as np
import glob 
from pathlib import Path 
import calendar
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm
from matplotlib.colors import from_levels_and_colors
def plot_weights(lat, lon, var, titulo, output):
    #funcion para graficar pesos (ergo barra entre 0 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    [dx,dy] = np.meshgrid (lon,lat)
    plt.figure()
    mapproj = bm.Basemap(projection='cyl', llcrnrlat=lats,
                    llcrnrlon= lonw, urcrnrlat= latn, urcrnrlon= lone)
    #projection and map limits
    mapproj.drawcoastlines(linewidth = 0.5)          # coast
    mapproj.drawcountries(linewidth = 0.1)         #countries
    lonproj, latproj = mapproj(dx, dy)      #poject grid
    # set desired contour levels.
    clevs = np.array([-0.3,-0.1,0.1,0.3,0.5])
#    barra = plt.cm.get_cmap('PRGn',5) #colorbar
    num_levels = 5
    vmin, vmax = -0.3,0.5
    midpoint = 0
    levels = np.linspace(vmin, vmax, num_levels)
    midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
    vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
    colors = plt.cm.PRGn(vals)
    cmap, norm = from_levels_and_colors(levels, colors)
#    fig, ax = plt.subplots()
#    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='none')
#    fig.colorbar(im)
    CS1 = mapproj.pcolor(lonproj, latproj, var, cmap = cmap, norm = norm, vmin = -0.3, vmax = 0.5)
    cbar = plt.colorbar(CS1, ticks = clevs)
    cbar.ax.tick_params(labelsize = 8)
    plt.title(titulo)
    plt.savefig(output, dpi=300, bbox_inches='tight', papertype='A4')
    plt.close()
    return

def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Plot models weights')
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable calibrated (prec or temp)')
    parser.add_argument('IC', type = int, nargs= 1,\
            help = 'Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type = int, nargs = 1,\
            help = 'Forecast leatime (in months, from 1 to 7)')
    parser.add_argument('mod_spread',  type = float, nargs = 1,\
            help = 'percentage of spread retained in each model (from 0 to 1)')
    parser.add_argument('--no-model', nargs = '+', choices = ['CFSv2','CanCM3','CanCM4',\
            'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'], dest ='no_model',\
            help = 'Models to be discarded')

    # Extract dates from args
    args=parser.parse_args()
    lista = glob.glob("/home/osman/proyectos/postdoc/modelos/*")
 
    if args.no_model is not None: #si tengo que descartar modelos
        lista = [i for i in lista if [line.rstrip('\n') for line in open(i)][0] not in args.no_model]
    
    keys = ['nombre', 'instit', 'latn', 'lonn', 'miembros', 'plazos', 'fechai', 'fechaf','ext']
    modelos = []

    for i in lista:
        lines = [line.rstrip('\n') for line in open(i)]
        modelos.append(dict(zip(keys, [lines[0], lines[1], lines[2], lines[3], int(lines[4]), 
            int(lines[5]), int(lines[6]), int(lines[7]), lines[8]])))

    nmodels = len(modelos)
    ny = int(np.abs(coords['lat_n'] - coords['lat_s']) + 1)
    nx = int(np.abs (coords['lon_e'] - coords['lon_w']) + 1) #doen not work if domain goes beyond greenwich
    nyears = int(modelos[0]['fechaf'] - modelos[0]['fechai'] + 1)
    weight = np.array([]).reshape(nyears, ny, nx, 0)
    rmean = np.array([]).reshape(ny, nx, 0)

    #defino ref dataset y target season
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i-12 if i>12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    
    for it in modelos:
        
        #abro archivo modelo
        archivo = Path('/datos/osman/nmme_output/cal_forecasts/'+ args.variable[0] + '_' + it['nombre'
            ]+'_' + calendar.month_abbr[args.IC[0]] +'_'+ SSS + '_gp_01_p_'+'{:03}'.format(args.mod_spread[0])+'_hind.npz')

        if archivo.is_file():
            data = np.load(archivo)
            
            #extraigo datos del peso de cada modelo.
            peso = data ['peso']
            weight =np.concatenate((weight,peso[:,0,:,:][:,:,:,np.newaxis]), axis = 3)
            peso = []
            Rm = data ['Rm']
                        
            rmean = np.concatenate ((rmean, Rm[:,:,np.newaxis]), axis = 2)
    
    lat = data ['lats']
    lon = data ['lons']
    #defino matriz de peso
    maximo = np.ndarray.argmax (weight, axis = 3) #posicion en donde se da el maximo
    weight = []
    peso = np.empty([ny,nx, nmodels])

    for i in np.arange(nmodels):
        peso [:,:,i] = np.sum (maximo == i, axis = 0)/nyears

    weight_pdf = peso # nlat nlon nmodels
    peso = []
    rmean [np.where(rmean < 0)] = 0
    rmean [np.sum(rmean, axis = 2)== 0,:] = 1
    weight_rmean = rmean / np.tile(np.sum(rmean, axis = 2)[:,:,np.newaxis],[1,1,nmodels]) #nlat nlon nmodels
    rmean = []
    
    #guardo los pronos
    ii = 0
    IC = calendar.month_abbr[args.IC[0]]
    for i in modelos:
        
        route = '/datos/osman/nmme_output/models_weight/'
        archivo = args.variable[0] + '_' + i['nombre'] +'_' + IC + '_' + SSS +\
                '_gp_01_p_' + '{:03}'.format(args.mod_spread[0]) + '_pdf_int_hind.npz'
        np.savez(route+archivo, peso = weight_pdf[:,:,ii], lat = lat, lon = lon)
        archivo = args.variable[0] + '_' + i['nombre']+ '_' + IC + '_' + SSS +\
                '_gp_01_p_rmean_hind.npz'
        np.savez(route+archivo, peso = weight_rmean[:,:,ii], lat = lat, lon = lon)

        route = '/datos/osman/nmme_figuras/models_weight/'
        archivo = args.variable[0] + '_' + i['nombre'] +'_' + IC + '_' + SSS +\
                '_gp_01_p_' + '{:03}'.format(args.mod_spread[0]) + '_pdf_int.png'
        
        plot_weights(lat, lon, weight_pdf[:,:,ii]-1/nmodels, 'Model Weight - ' + i['nombre'], route+archivo)

        archivo = args.variable[0] + '_' + i['nombre']+ '_' + IC + '_' + SSS +\
                '_gp_01_p_rmean.png'
        plot_weights(lat, lon, weight_rmean[:,:,ii]-1/nmodels, 'Model Weight - ' + i['nombre'], route+archivo)
        ii = ii + 1

#===================================================================================================

start = time.time()

#abro archivo donde guardo coordenadas
                                                 
coordenadas = 'coords'

lines = [line.rstrip('\n') for line in open(coordenadas)]

coords = {'lat_s' : float(lines[1]),
        'lat_n' : float(lines [2]),
        'lon_w' : float(lines[3]),
        'lon_e' : float(lines[4])}

main()

end = time.time()

print(end - start)

# =================================================================================
