#grafico pronostico
import argparse #parse command line options
import time #test time consummed
import calendar
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

def manipular_nc(archivo, variable, lat_name, lon_name, lats, latn, lonw, lone):
    """gets netdf variables"""
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
def plot_pronosticos(pronos, dx, dy, titulo, salida):
    """Plot probabilistic forecast"""
    fig = plt.figure()
    mapproj = bm.Basemap(projection='cyl', llcrnrlat=lats,
                         llcrnrlon=lonw, urcrnrlat=latn, urcrnrlon=lone)
    #projection and map limits
    mapproj.drawcoastlines()          # coast
    mapproj.drawcountries()         #countries
    lonproj, latproj = mapproj(dx, dy)      #poject grid
    CS1 = mapproj.pcolor(lonproj, latproj, pronos, cmap=cmap, vmin=0.5, vmax=12.5)
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
    cb3.set_ticklabels(['45%','55%','65%','+70%'])
    cb3.ax.tick_params(labelsize=7)
    cb3.set_label('Upper')
    plt.savefig(salida, dpi=600, bbox_inches='tight', papertype='A4')
    plt.close()
    return
def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Verify combined forecast')
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable to verify (prec or temp)')
    parser.add_argument('IC', type = int, nargs= 1,\
            help = 'Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type = int, nargs = 1,\
            help = 'Forecast leatime (in months, from 1 to 7)')

    args=parser.parse_args()
    #defino ref dataset y target season
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)

    wtech = ['pdf_int', 'mean_cor', 'same']
    ctech = ['wpdf', 'wsereg']
    #genero barra de colores
    colores = np.array([[166., 54., 3.], [230., 85., 13.], [253., 141., 60.],
                        [253., 190., 133.], [227., 227., 227.], [204., 204.,
                                                                 204.],
                        [150., 150., 150.], [82., 82., 82.], [186., 228.,
                                                              179.],
                        [116., 196., 118.], [49., 163., 84.], [0., 109.,
                                                               44.]]) / 255
    cmap = mpl.colors.ListedColormap(colores)
    #open and handle land-sea mask
    lsmask = "/datos/osman/nmme/monthly/lsmask.nc"
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
    ruta = '/datos/osman/nmme_output/comb_forecast/'
    for i in ctech:
        for j in wtech:
            archivo = args.variable[0] + '_mme_' + calendar.month_abbr[
                args.IC[0]] +'_' + SSS + '_' + j + '_' + i + '_hind.npz'
            data = np.load(ruta + archivo)
            lat = data['lat']
            lon = data['lon']
            lats = np.min(lat)
            latn = np.max(lat)
            lonw = np.min(lon)
            lone = np.max(lon)
            [dx, dy] = np.meshgrid(lon, lat)
            ruta = '/datos/osman/nmme_figuras/forecast/'
            for k in np.arange(year_verif, 2011, 1):
                output = ruta + 'for_prec_' + SSS + '_ic_' + \
                        calendar.month_abbr[args.IC[0]] + '_' + str(k) + '_' +\
                        i + '_' + j + '.png'
                for_terciles = np.squeeze(data['prob_terc_comb'][:, k - 1982, :, :])
                #agrego el prono de la categoria above normal
                below = ndimage.filters.gaussian_filter(for_terciles[0, :,
                                                                     :], 1,
                                                        order=0, output=None,
                                                        mode='reflect')
                near = ndimage.filters.gaussian_filter(for_terciles[1, :, :]\
                                                             - for_terciles[0, :,\
                                                                            :], 1,
                                                       order=0, output=None,
                                                       mode='reflect')
                above = ndimage.filters.gaussian_filter(1 - for_terciles[1, :,
                                                                         :], 1,
                                                        order=0, output=None,
                                                        mode='reflect')
                for_terciles = np.concatenate([below[:, :, np.newaxis],
                                               near[:, :, np.newaxis],
                                               above[:, :, np.newaxis]], axis=2)
                for_mask = asignar_categoria(for_terciles)
                for_mask = np.ma.masked_array(for_mask,
                                              np.logical_not(land.astype(bool)))
                plot_pronosticos(for_mask, dx, dy, SSS + ' Forecast '\
                                 'IC ' + calendar.month_abbr[args.IC[0]] +\
                                 + '_' + str(k) + ' - ' + i + '-' + j, output)

    ruta = '/datos/osman/nmme_output/comb_forecast/'
    archivo = args.variable[0] + '_mme_' + calendar.month_abbr[args.IC[0]] +\
            '_'Nov_DJF_gp_01_p_1.0_same_count_hind.npz'
    data = np.load(ruta + archivo)
    lat = data['lat']
    lon = data['lon']
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    [dx,dy] = np.meshgrid (lon, lat)
    ruta = '/datos/osman/nmme_figuras/forecast/'
    for k in np.arange(1982, 2010, 1):

        output = ruta + 'for_prec_DJF_ic_nov_' + str(k) + '_count.png'
        for_terciles = np.squeeze(data['prob_terc_comb'][:, k-1982, :, :])
        #agrego el prono de la categoria above normal
        for_terciles = np.concatenate([for_terciles[0, :, :][:, :, np.newaxis],
                                       (for_terciles[1, :, :] -
                                        for_terciles[0, :, :])[:, :,
                                                               np.newaxis],
                                       (1 - for_terciles[1, :, :])[:, :,
                                                                   np.newaxis]],
                                      axis=2)
        for_mask = asignar_categoria(for_terciles)
        for_mask = np.ma.masked_array(for_mask,
                                      np.logical_not(land.astype(bool)))
        plot_pronosticos(for_mask, dx, dy, 'DJF Precipitation Forecast IC Nov. ' +
                         str(k) + ' - Uncalibrated', output)
#===================================================================================================
start = time.time()
main()
end = time.time()
print(end - start)


