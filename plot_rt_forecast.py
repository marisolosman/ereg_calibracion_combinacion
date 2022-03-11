"""smoth calibrated probabilities using a gaussian filter and plot forecast"""
import argparse  # parse command line options
import time  # test time consummed
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
### Modificado M
mpl.use('agg')
###
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import configuration
import os

cfg = configuration.Config.Instance()

def manipular_nc(archivo, variable, lat_name, lon_name, lats, latn, lonw, lone):
    """gets netdf variables"""
    #reportar lectura de un archivo descargado
    cfg.report_input_file_used(archivo)
    #continuar ejecución
    dataset = xr.open_dataset(archivo, decode_times=False)
    var_out = dataset[variable].sel(**{lat_name: slice(lats, latn), lon_name: slice(lonw, lone)})
    lon = dataset[lon_name].sel(**{lon_name: slice(lonw, lone)})
    lat = dataset[lat_name].sel(**{lat_name: slice(lats, latn)})
    return var_out, lat, lon

def asignar_categoria(for_terciles):
    """determines most likely category"""
    most_likely_cat = np.argmax(for_terciles, axis=2)
    [nlats, nlons] = for_terciles.shape[0:2]
    for_cat = np.zeros([nlats, nlons], dtype=int)
    for_cat.fill(np.nan)
    for ii in np.arange(nlats):
        for jj in np.arange(nlons):
            if (most_likely_cat[ii, jj] == 2):
                if for_terciles[ii, jj, 2] >= 0.7:
                    for_cat[ii, jj] = 12
                elif for_terciles[ii, jj, 2] >= 0.6:
                    for_cat[ii, jj] = 11
                elif for_terciles[ii, jj, 2] >= 0.5:
                    for_cat[ii, jj] = 10
                elif for_terciles[ii, jj, 2] >= 0.4:
                    for_cat[ii, jj] = 9
            elif (most_likely_cat[ii, jj] == 0):
                if for_terciles[ii, jj, 0] >= 0.7:
                    for_cat[ii, jj] = 1
                elif for_terciles[ii, jj, 0] >= 0.6:
                    for_cat[ii, jj] = 2
                elif for_terciles[ii, jj, 0] >= 0.5:
                    for_cat[ii, jj] = 3
                elif for_terciles[ii, jj, 0] >= 0.4:
                    for_cat[ii, jj] = 4
            elif (most_likely_cat[ii, jj] == 1):
                if for_terciles[ii, jj, 1] >= 0.7:
                    for_cat[ii, jj] = 8
                elif for_terciles[ii, jj, 1] >= 0.6:
                    for_cat[ii, jj] = 7
                elif for_terciles[ii, jj, 1] >= 0.5:
                    for_cat[ii, jj] = 6
                elif for_terciles[ii, jj, 1] >= 0.4:
                    for_cat[ii, jj] = 5

            mascara = for_cat < 1
            for_mask = np.ma.masked_array(for_cat, mascara)
    return for_mask
def plot_pronosticos(pronos, dx, dy, lats, latn, lonw, lone, cmap, colores, vmin, vmax,
                     titulo, salida):
    """Plot probabilistic forecast"""
    init_message = f"Generating figure: {os.path.basename(salida)}"
    print(init_message) if not cfg.get('use_logger') else cfg.logger.info(init_message)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure()
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2)
    ax = plt.axes(projection=mapproj)
    ax.set_extent(limits, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5, resolution='50m')
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    CS1 = ax.pcolor(dx, dy, pronos, cmap=cmap, vmin=vmin, vmax=vmax,
                   transform=ccrs.PlateCarree())
    #ax.pcolor(dx, dy, pronos, cmap='coral', vmin=90, vmax=90, transform=ccrs.PlateCarree())
    #genero colorbar para pronos
    plt.title(titulo)
    ax1 = fig.add_axes([0.2, 0.05, 0.2, 0.03])
    cmap1 = mpl.colors.ListedColormap(colores[0:4, :])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap1, norm=norm, boundaries=bounds,
                                    ticks=[1, 2, 3, 4], spacing='uniform',
                                    orientation='horizontal')
    cb1.set_ticklabels(['+70%', '65%', '55%', '45%'])
    cb1.ax.tick_params(labelsize=7)
    cb1.set_label('Lower')
    ax2 = fig.add_axes([0.415, 0.05, 0.2, 0.03])
    cmap2 = mpl.colors.ListedColormap(colores[4:8, :])
    bounds = [4.5, 5.5, 6.5, 7.5, 8.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap2, norm=norm, boundaries=bounds,
                                    ticks=[5, 6, 7, 8], spacing='uniform',
                                    orientation='horizontal')
    cb2.set_ticklabels(['45%', '55%', '65%', '+70%'])
    cb2.ax.tick_params(labelsize=7)
    cb2.set_label('Normal')
    ax3 = fig.add_axes([0.63, 0.05, 0.2, 0.03])
    cmap3 = mpl.colors.ListedColormap(colores[8:, :])
    bounds = [8.5, 9.5, 10.5, 11.5, 12.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap3, norm=norm, boundaries=bounds,
                                    ticks=[9, 10, 11, 12], spacing='uniform',
                                    orientation='horizontal')
    cb3.set_ticklabels(['45%', '55%', '65%', '+70%'])
    cb3.ax.tick_params(labelsize=7)
    cb3.set_label('Upper')
    plt.savefig(salida, dpi=600, bbox_inches='tight', papertype='A4')
    plt.close()
    cfg.set_correct_group_to_file(salida)  # Change group of file
    saved_message = f"Saved figure: {os.path.basename(salida)}"
    print(saved_message) if not cfg.get('use_logger') else cfg.logger.info(saved_message)
    return

def main(args):
    # Defino ref dataset y target season
    initialDate = datetime.datetime.strptime(args.IC[0], '%Y-%m-%d')
    iniy = initialDate.year
    inim = initialDate.month
    INIM = calendar.month_abbr[inim]
    seas = range(inim + args.leadtime[0], inim + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)

    wtech = args.weighting  # ['pdf_int', 'mean_cor', 'same']
    ctech = args.combination  # ['wpdf', 'wsereg']
    #custom colorbar
    if args.variable[0] == 'prec':
        colores = np.array([[166., 54., 3.], [230., 85., 13.], [253., 141.,
                                                                60.],
                            [253., 190., 133.], [227., 227., 227.],
                            [204., 204., 204.], [150., 150., 150.],
                            [82., 82., 82.], [186., 228., 179.],
                            [116., 196., 118.], [49., 163., 84.],
                            [0., 109., 44.], [241., 233., 218.]]) / 255
        vmin = 0.5
        vmax = 13.5
        PATH = cfg.get('folders').get('download_folder')
        drymask = Path(PATH, cfg.get('folders').get('nmme').get('root'), 'dry_mask.nc')
        dms = xr.open_dataset(drymask)
        #selecciono mascara del mes
        dms = dms.sel(month=sss[1])

    else:
        colores = np.array([[8., 81., 156.], [49., 130., 189.],
                            [107., 174., 214.], [189., 215., 231.],
                            [227., 227., 227.], [204., 204., 204.],
                            [150., 150., 150.], [82., 82., 82.],
                            [252., 174., 145.], [251., 106., 74.],
                            [222., 45., 38.], [165., 15., 21.]]) / 255
        vmin = 0.5
        vmax = 12.5

    cmap = mpl.colors.ListedColormap(colores)
    #open and handle land-sea mask
    PATH = cfg.get('folders').get('download_folder')
    lsmask = Path(PATH, cfg.get('folders').get('nmme').get('root'), 'lsmask.nc')
    coords = cfg.get('coords')
    [land, Y, X] = manipular_nc(lsmask, "land", "Y", "X", coords['lat_n'],
                                coords['lat_s'], coords['lon_w'],
                                coords['lon_e'])
    land = np.flipud(land)
    PATH = cfg.get('folders').get('gen_data_folder')
    RUTA = Path(PATH, cfg.get('folders').get('data').get('real_time_forecasts'))
    RUTA_IM = Path(PATH, cfg.get('folders').get('figures').get('real_time_forecasts'))
    for i in ctech:
        for j in wtech:
            archivo = Path(RUTA, args.variable[0] + '_mme_' + INIM + str(iniy)\
                           + '_' + SSS + '_gp_01_' + j + '_' + i + '.npz')
            data = np.load(archivo)
            lat = data['lat']
            lon = data['lon']
            lats = np.min(lat)
            latn = np.max(lat)
            lonw = np.min(lon)
            lone = np.max(lon)
            [dx, dy] = np.meshgrid(lon, lat)
            output = Path(RUTA_IM, 'for_' + args.variable[0] + '_' + SSS + '_ic_' +
                          INIM + '_' + str(iniy) + '_' + i + '_' + j + '.png')
            for_terciles = np.squeeze(data['prob_terc_comb'][:, :, :])
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
            ### Modificado M
            message = f"max: {np.nanmax(for_terciles)}, min: {np.nanmin(for_terciles)}"
            print(message) if not cfg.get('use_logger') else cfg.logger.info(message)
            ###
            for_mask = asignar_categoria(for_terciles)
            if args.variable[0] =='prec':
                for_mask[dms.prec.values] = 13

            for_mask = np.ma.masked_array(for_mask,
                                          np.logical_not(land.astype(bool)))
            plot_pronosticos(for_mask, dx, dy, lats, latn, lonw, lone,
                             cmap, colores, vmin, vmax, SSS + ' ' + args.variable[0] +\
                             ' Forecast IC ' + INIM + ' ' + str(iniy) +\
                             ' - ' + i + '-' + j, output)


# ==================================================================================================
if __name__ == "__main__":
  
    # Define parser data
    parser = argparse.ArgumentParser(description='Verify combined forecast')
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable to verify (prec or tref)')
    parser.add_argument('--IC', type=str, nargs=1,\
            help='Date of initial conditions (in "YYYY-MM-DD")')
    parser.add_argument('--leadtime', type=int, nargs=1,\
            help='Forecast leatime (in months, from 1 to 7)')
    parser.add_argument('--weighting', nargs='+',
            default=["same", "pdf_int", "mean_cor"], choices=["same", "pdf_int", "mean_cor"],
            help='Weighting methods to be plotted.')
    parser.add_argument('--combination', nargs='+',
            default=["wpdf", "wsereg"], choices=["wpdf", "wsereg"],
            help='Combination methods to be plotted.')

    # Extract data from args
    args = parser.parse_args()
  
    # Run plotting
    start = time.time()
    try:
        main(args)
    except Exception as e:
        error_detected = True
        cfg.logger.error(f"Failed to run \"plot_rt_forecast.py\". Error: {e}.")
        raise  # see: http://www.markbetz.net/2014/04/30/re-raising-exceptions-in-python/
    else:
        error_detected = False
    finally:
        end = time.time()
        err_pfx = "with" if error_detected else "without"
        message = f"Total time to run \"plot_rt_forecast.py\" ({err_pfx} errors): {end - start}" 
        print(message) if not cfg.get('use_logger') else cfg.logger.info(message)


