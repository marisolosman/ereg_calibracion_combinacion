#grafico pronostico
import argparse #para hacer el llamado desde consola
import time
import calendar
import numpy as np
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

def main():
    """Plot observed category"""
    # Define parser data
    parser = argparse.ArgumentParser(description='Plot observed category')
    parser.add_argument('variable',type=str, nargs=1,\
            help='Variable to verify (prec or temp)')
    parser.add_argument('IC', type=int, nargs=1,\
            help='Month of beginning of season (from 1 for Jan to 12 for Dec)')
    args=parser.parse_args()

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

    #defino ref dataset y target season
    seas = range(args.IC[0], args.IC[0] + 3)
    sss = [i-12 if i>12 else i for i in seas]
    year_verif = 1982 if (seas[0] > 1 and seas[0] < 11) else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    #obtengo datos observados
    RUTA = '/datos/osman/nmme_output/'
    RUTA_FIG = '/datos/osman/nmme_figuras/forecast/'
    archivo = 'obs_' + args.variable[0] + '_' + str(year_verif) + '_' + SSS +\
            '.npz'
    data = np.load(RUTA + archivo)
    obs_terciles = data['cat_obs']
    nlats = obs_terciles.shape[2]
    nlons = obs_terciles.shape[3]
    lat = data['lats_obs']
    lon = data['lons_obs']
    lats = np.min(lat)
    obs = np.zeros([nlats, nlons])
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    [dx,dy] = np.meshgrid(lon,lat)
    cmap = mpl.colors.ListedColormap(np.array([[217, 95, 14], [189, 189, 189],
                                               [44, 162, 95]]) / 256)
    for k in np.arange(year_verif, 2011):
        output = RUTA_FIG + 'obs_' + args.variable[0] + '_' + SSS + '_' + str(k)\
                + '.png'
        obs_cat = obs_terciles[:, k - year_verif, :, :]
        obs[obs_cat[1, :, :] == 1] = 1
        obs[obs_cat[2, :, :] == 1] = 2
        obs[obs_cat[0, :, :] == 1] = 0
        obs = np.ma.masked_array(obs, np.logical_not(land.astype(bool)))
        fig = plt.figure()
        mapproj = bm.Basemap(projection='cyl', llcrnrlat=lats, llcrnrlon=lonw,
                             urcrnrlat=latn, urcrnrlon=lone)
        #projection and map limits
        mapproj.drawcoastlines()          # coast
        mapproj.drawcountries()         #countries
        lonproj, latproj = mapproj(dx, dy)      #poject grid
        CS1 = mapproj.pcolor(lonproj, latproj, obs, cmap=cmap, alpha=0.6,
                             vmin=-0.5, vmax=2.5)
        plt.title(SSS + ' Observed Category - ' + str(k))
        ax = fig.add_axes([0.42, 0.05, 0.2, 0.03])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                       boundaries=bounds, ticks=[0, 1, 2],
                                       spacing='uniform',
                                       orientation='horizontal', alpha=0.6)
        cb.set_ticklabels(['Lower', 'Middle', 'Upper'])
        cb.ax.tick_params(labelsize=7)
        plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
        plt.close()
#==========================================================================
start = time.time()
main()
end = time.time()
print(end - start)
