"""smoth calibrated probabilities using a gaussian filter and plot forecast"""
import argparse #parse command line options
import time #test time consummed
import calendar
import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import scipy.ndimage as ndimage
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
def manipular_nc(archivo, variable, lat_name, lon_name, lats, latn, lonw, lone):
    """gets netdf variables"""
    dataset = xr.open_dataset(archivo, decode_times=False)
    var_out = dataset[variable].sel(**{lat_name: slice(lats, latn), lon_name: slice(lonw, lone)})
    lon = dataset[lon_name].sel(**{lon_name: slice(lonw, lone)})
    lat = dataset[lat_name].sel(**{lat_name: slice(lats, latn)})
    return var_out, lat, lon

def asignar_categoria(for_terciles):
    """determines most likely category"""
    for_cat = for_terciles * 100
    mascara = for_cat < 10
    for_mask = np.ma.masked_array(for_cat, mascara)
    return for_mask
def plot_pronosticos(pronos, dx, dy, lats, latn, lonw, lone, cmap, colores,
                     titulo, salida):
    """Plot probabilistic forecast"""
    limits = [lonw, lone, lats, -10]
    fig = plt.figure()
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2)
    ax = plt.axes(projection=mapproj)
    ax.set_extent(limits, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5, resolution='50m')
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    CS1 = ax.pcolor(dx, dy, pronos, cmap=cmap,
                   transform=ccrs.PlateCarree())
    #ax.pcolor(dx, dy, pronos, cmap='coral', vmin=90, vmax=90, transform=ccrs.PlateCarree())
    #genero colorbar para pronos
    plt.title(titulo)
    ax1 = fig.add_axes([0.35, 0.05, 0.3, 0.03])
#    cmap1 = mpl.colors.ListedColormap(colores[0:4, :])
#    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
#    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=colores,
                                    orientation='horizontal')
    plt.savefig(salida, dpi=600, bbox_inches='tight', papertype='A4')
    plt.close()
    return

def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Plot combined forecast')
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable to verify (prec or temp)')
    parser.add_argument('--IC', type=str, nargs=1,\
            help='Date of initial conditions (in "YYYY-MM-DD")')
    parser.add_argument('--leadtime', type=int, nargs=1,\
            help='Forecast leatime (in months, from 1 to 7)')

    args=parser.parse_args()
    #defino ref dataset y target season
    initialDate = datetime.datetime.strptime(args.IC[0], '%Y-%m-%d')
    iniy = initialDate.year
    inim = initialDate.month
    INIM = calendar.month_abbr[inim]
    seas = range(inim + args.leadtime[0], inim + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)

    wtech = ['mean_cor']
    ctech = ['wsereg']
    #custom colorbar
    file1 = open("configuracion", 'r')
    PATH = file1.readline().rstrip('\n')
    file1.close()
    for pp in ['quintiles', 'terciles']:
        if pp == 'quintiles':
            bounds = np.arange(10, 90, 10)
            prob_low = '20%'
            prob_high = '80%'
            nombres = '_extremes_mme_'
            sub_var = 'prob_quint_comb'
        else:
            bounds= [0, 33, 50, 66, 100]
            prob_low = '33%'
            prob_high = '66%'
            nombres = '_mme_'
            sub_var = 'prob_terc_comb'

        if args.variable[0] == 'prec':
            colores_above = mpl.cm.Greens
            colores_below = mpl.cm.Oranges
#            bounds = np.arange(10, 90, 10)
            drymask = PATH + 'DATA/dry_mask.nc'
            dms = xr.open_dataset(drymask)
            #selecciono mascara del mes
            dms = dms.sel(month=sss[1])

        else:
            colores_above = mpl.cm.Reds
            colores_below = mpl.cm.Blues
#            bounds = np.arange(10, 90, 10)
        #open and handle land-sea mask
        lsmask = PATH + "NMME/lsmask.nc"
        coordenadas = 'coords'
        domain = [line.rstrip('\n') for line in open(coordenadas)]  #Get domain limits
        coords = {'lat_s': float(domain[1]),
                  'lat_n': float(domain[2]),
                  'lon_w': float(domain[3]),
                  'lon_e': float(domain[4])}
        [land, Y, X] = manipular_nc(lsmask, "land", "Y", "X", coords['lat_n'],
                                    coords['lat_s'], coords['lon_w'],
                                    coords['lon_e'])
        land = np.flipud(land)
        RUTA = PATH + 'DATA/real_time_forecasts/'
        RUTA_IM = PATH + 'FIGURES/'
        for i in ctech:
            for j in wtech:
                archivo = Path(RUTA + args.variable[0] + nombres + INIM + str(iniy)\
                               + '_' + SSS + '_gp_01_' + j + '_' + i + '.npz')
                data = np.load(archivo)
                lat = data['lat']
                lon = data['lon']
                lats = np.min(lat)
                latn = np.max(lat)
                lonw = np.min(lon)
                lone = np.max(lon)
                [dx, dy] = np.meshgrid(lon, lat)

                for_terciles = np.squeeze(data[sub_var][:, :, :])
                #agrego el prono de la categoria above normal
                if args.variable[0] == 'prec':
                    below = ndimage.filters.gaussian_filter(for_terciles[0, :,
                                                                         :], 1,
                                                            order=0, output=None,
                                                            mode='constant')
                    near = ndimage.filters.gaussian_filter(for_terciles[1, :,
                                                                             :], 1,
                                                            order=0, output=None,
                                                            mode='constant')
                else:
                    for_terciles[:, for_terciles[1, :, :] == 0] = np.nan
                    kernel = Gaussian2DKernel(x_stddev=1)
                    below = convolve(for_terciles[0, :, :], kernel, preserve_nan=True)
                    near = convolve(for_terciles[1, :, :], kernel, preserve_nan=True)
                above = 1 - near
                near = near - below
                for_terciles = np.concatenate([below[:, :, np.newaxis],
                                               near[:, :, np.newaxis],
                                               above[:, :, np.newaxis]], axis=2)
                for_mask = asignar_categoria(for_terciles)
                
                if args.variable[0] =='prec':
                    for_mask[dms.prec.values, :] = 0
                    # plot below
                    output = RUTA_IM + 'for_' + args.variable[0] + '_low' + prob_low + '_' + SSS + '_ic_'\
                            + INIM + '_' + str(iniy) + '.png'

                    plot_pronosticos(np.ma.masked_array(for_mask[:, :, 0],
                                                        np.logical_not(land.astype(bool))),
                                     dx, dy, lats, latn, lonw, lone,
                                     colores_below,
                                     mpl.colors.BoundaryNorm(bounds, colores_below.N), 
                                     SSS + ' ' + args.variable[0] +\
                                     ' below ' + prob_low + ' hist range -  Forecast IC ' + INIM + ' ' + str(iniy),
                                     output)
                    output = RUTA_IM + 'for_' + args.variable[0] + '_high' + prob_high + '_' + SSS + '_ic_'\
                            + INIM + '_' + str(iniy) + '.png'

                    plot_pronosticos(np.ma.masked_array(for_mask[:, :, 2],
                                                        np.logical_not(land.astype(bool))),
                                     dx, dy, lats, latn, lonw, lone,
                                     colores_above,
                                     mpl.colors.BoundaryNorm(bounds, colores_above.N), 
                                     SSS + ' ' + args.variable[0] +\
                                     ' above ' + prob_high + ' hist range -  Forecast IC ' + INIM + ' ' + str(iniy),
                                     output)

                else:
                    output = RUTA_IM + 'for_' + args.variable[0] + '_high' + prob_high + '_' + SSS + '_ic_'\
                            + INIM + '_' + str(iniy) + '.png'

                    plot_pronosticos(np.ma.masked_array(for_mask[:, :, 2],
                                                        np.logical_not(land.astype(bool))),
                                     dx, dy, lats, latn, lonw, lone,
                                     colores_above,
                                     mpl.colors.BoundaryNorm(bounds, colores_above.N), 
                                     SSS + ' ' + args.variable[0] +\
                                     ' above ' + prob_high + ' hist range -  Forecast IC ' + INIM + ' ' + str(iniy),
                                     output)
                    output = RUTA_IM + 'for_' + args.variable[0] + '_low' + prob_low + '_' + SSS + '_ic_'\
                            + INIM + '_' + str(iniy) + '.png'

                    plot_pronosticos(np.ma.masked_array(for_mask[:, :, 0],
                                                        np.logical_not(land.astype(bool))),
                                     dx, dy, lats, latn, lonw, lone,
                                     colores_below,
                                     mpl.colors.BoundaryNorm(bounds, colores_below.N), 
                                     SSS + ' ' + args.variable[0] +\
                                     ' below ' + prob_low + ' hist range -  Forecast IC ' + INIM + ' ' + str(iniy),
                                     output)

#================================================================================================
start = time.time()
main()
end = time.time()
print(end - start)


