#grafico pronostico
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm
from matplotlib.colors import LinearSegmentedColormap

#tomo prono de DJF con CI en Nov 1997
wtech = ['pdf_int', 'mean_cor', 'same']
ctech = ['wpdf', 'wsereg']
#abro pronosticos
colores = np.array([[166.,54.,3.], [230.,85.,13.], [253.,141.,60.], [253.,190.,133.], [227.,227.,227.], [204.,204.,204.], [150.,150.,150.],[82.,82.,82.], 
    [186.,228.,179.],[116.,196.,118.],[49.,163.,84.],[0.,109.,44.]])/255

#cmap = mpl.colors.ListedColormap(colores)
#for i in ctech:
#    for j in wtech:
#        if i == 'wsereg':
#            archivo = 'prec_mme_Nov_DJF_gp_01_p_1.0_'+ j +'_'+ i + '_p_1.0_hind.npz'
#        else:
#            archivo = 'prec_mme_Nov_DJF_gp_01_p_1.0_'+ j +'_'+ i + '_hind.npz'
#        ruta = '/datos/osman/nmme_output/comb_forecast/'
#        data = np.load(ruta + archivo)
#        lat = data['lat']
#        lon = data['lon']
#        lats = np.min(lat)
#        latn = np.max(lat)
#        lonw = np.min(lon)
#        lone = np.max(lon)
#        [dx,dy] = np.meshgrid (lon,lat)
#
#        for k in np.arange(1982,2010,1):
#            output = '/datos/osman/nmme_figuras/for_prec_DJF_ic_nov_' + str(k) + '_'+ i + '_' + j + '.png'
#          
#            for_terciles = np.squeeze(data['prob_terc_comb'][:,k-1982,:,:])
#            #agrego el prono de la categoria above normal
#            for_terciles = np.concatenate([for_terciles[0,:,:][:,:,np.newaxis],(for_terciles[1,:,:]-\
#                    for_terciles[0,:,:])[:,:,np.newaxis],(1-for_terciles[1,:,:])[:,:,np.newaxis]], axis=2)
#    #
#    #        smooth_forecast = np.array(np.shape(for_terciles))
#    #
#    #        for i in np.arange(np.shape(for_terciles)[0]):
#    #            for j in np.arange(np.shape(for_terciles)[1]):
#    #                print(i,j)
#    #                if ((i >=1) & (i<(np.shape(for_terciles)[0]-1))) &  ((j >=1) & (j<(np.shape(for_terciles)[1]-1))):
#    #                    smooth_forecast[i,j,:] = np.nanmean(np.reshape(for_terciles[i-1:i+2,j-1:j+2,:],[9,3]), axis = 1)
#    #                elif i==0: #faltan casos!!
#    #                    smooth_forecast[i,j,:] = np.nanmean(np.reshape(for_terciles[i:i+2,j-1:j+2,:],[6,3]), axis = 1)
#    #                elif i==(np.shape(for_terciles)[0]-1):
#    #                    smooth_forecast[i,j,:] = np.nanmean(np.reshape(for_terciles[i-1:,j-1:j+2,:],[6,3]), axis = 1)
#    #                elif j==0:
#    #                    smooth_forecast[i,j,:] = np.nanmean(np.reshape(for_terciles[i-1:i+2,j:j+2,:],[6,3]), axis = 1)
#    #                else:
#    #                    smooth_forecast[i,j,:] = np.nanmean(np.reshape(for_terciles[i-1:i+2,j-1:,:],[6,3]), axis = 1)
#    #
#    #        for_terciles = smooth_forecast
#            #clasifico los pronos
#            #below normal mayor de 70 asigno 1 entre 70 y 60 2 y asi hasta 33y 40
#            most_likely_cat = np.argmax(for_terciles, axis = 2)
#            [nlats, nlons] = for_terciles.shape[0:2]
#            for_cat = np.zeros([nlats,nlons],dtype=int)
#            for_cat.fill(np.nan)
#            for ii in np.arange(nlats):
#                for jj in np.arange(nlons):
#                    if most_likely_cat[ii,jj]>0:
#                        if for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.7:
#                                for_cat[ii,jj] = 4 + most_likely_cat[ii,jj]*4
#                        elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.6:
#                            for_cat[ii,jj] = 3 + most_likely_cat[ii,jj]*4
#                        elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.5:
#                            for_cat[ii,jj] = 2 + most_likely_cat[ii,jj]*4
#                        elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.4:
#                            for_cat[ii,jj] = 1 + most_likely_cat[ii,jj]*4
#                    else:
#                        if for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.7:
#                            for_cat[ii,jj] = 1 + most_likely_cat[ii,jj]*4
#                        elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.6:
#                            for_cat[ii,jj] = 2 + most_likely_cat[ii,jj]*4
#                        elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.5:
#                            for_cat[ii,jj] = 3 + most_likely_cat[ii,jj]*4
#                        elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.4:
#                            for_cat[ii,jj] = 4 + most_likely_cat[ii,jj]*4
#
#            mascara = for_cat<1
#
#            for_mask = np.ma.masked_array(for_cat,mascara)
#            fig = plt.figure()
#            mapproj = bm.Basemap(projection='cyl', llcrnrlat=lats,
#            llcrnrlon= lonw, urcrnrlat= latn, urcrnrlon= lone)
#            #projection and map limits
#            mapproj.drawcoastlines()          # coast
#            mapproj.drawcountries()         #countries
#            lonproj, latproj = mapproj(dx, dy)      #poject grid
#            
#            CS1 = mapproj.pcolor(lonproj, latproj, for_mask, cmap = cmap, vmin = 0.5, vmax = 12.5)
#            #genero colorbar para pronos
#            plt.title('DJF Precipitation Forecast IC Nov. ' + str(k) +' - ' + i + '-' + j)
#            ax1 = fig.add_axes([0.2, 0.05, 0.2, 0.03])
#            cmap1 = mpl.colors.ListedColormap(colores[0:4,:])
#            bounds = [0.5,1.5,2.5,3.5,4.5]
#            norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
#            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap1, norm=norm, boundaries=bounds,
#                    ticks=[1,2,3,4],spacing='uniform', orientation='horizontal')
#            cb1.set_ticklabels(['+70%','65%','55%','45%'])
#            cb1.ax.tick_params(labelsize = 7)
#            cb1.set_label('Lower')
#            ax2 = fig.add_axes([0.415, 0.05, 0.2, 0.03])
#            cmap2 = mpl.colors.ListedColormap(colores[4:8,:])
#            bounds = [4.5,5.5,6.5,7.5,8.5]
#            norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
#            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap2, norm=norm, boundaries=bounds,
#                    ticks=[5,6,7,8],spacing='uniform', orientation='horizontal')
#            cb2.set_ticklabels(['45%','55%','65%','+70%'])
#            cb2.ax.tick_params(labelsize = 7)
#            cb2.set_label('Normal')
#            ax3 = fig.add_axes([0.63, 0.05, 0.2, 0.03])
#            cmap3 = mpl.colors.ListedColormap(colores[8:,:])
#            bounds = [8.5,9.5,10.5,11.5,12.5]
#            norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
#            cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap3, norm=norm, boundaries=bounds,
#                    ticks=[9,10,11,12], spacing='uniform', orientation='horizontal')
#            cb3.set_ticklabels(['45%','55%','65%','+70%'])
#            cb3.ax.tick_params(labelsize = 7)
#            cb3.set_label('Upper')
#            plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
#            plt.close()
#ruta = '/datos/osman/nmme_output/comb_forecast/'
#archivo = 'prec_mme_Nov_DJF_gp_01_p_1.0_same_count_hind.npz'
#data = np.load(ruta + archivo)
#lat = data['lat']
#lon = data['lon']
#lats = np.min(lat)
#latn = np.max(lat)
#lonw = np.min(lon)
#lone = np.max(lon)
#[dx,dy] = np.meshgrid (lon,lat)
#
##for k in np.arange(1982,2010,1):
#    output = '/datos/osman/nmme_figuras/for_prec_DJF_ic_nov_' + str(k) + '_count.png'
#    for_terciles = np.squeeze(data['prob_terc_comb'][:,k-1982,:,:])
#    #agrego el prono de la categoria above normal
#    for_terciles = np.concatenate([for_terciles[0,:,:][:,:,np.newaxis],(for_terciles[1,:,:]-\
#    for_terciles[0,:,:])[:,:,np.newaxis],(1-for_terciles[1,:,:])[:,:,np.newaxis]], axis=2)
#
#    #clasifico los pronos
#    #below normal mayor de 70 asigno 1 entre 70 y 60 2 y asi hasta 33y 40
#    most_likely_cat = np.argmax(for_terciles, axis = 2)
#    [nlats, nlons] = for_terciles.shape[0:2]
#    for_cat = np.zeros([nlats,nlons],dtype=int)
#    for_cat.fill(np.nan)
#    for ii in np.arange(nlats):
#        for jj in np.arange(nlons):
#                if most_likely_cat[ii,jj]>0:
#                    if for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.7:
#                            for_cat[ii,jj] = 4 + most_likely_cat[ii,jj]*4
#                    elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.6:
#                        for_cat[ii,jj] = 3 + most_likely_cat[ii,jj]*4
#                    elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.5:
#                        for_cat[ii,jj] = 2 + most_likely_cat[ii,jj]*4
#                    elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.4:
#                        for_cat[ii,jj] = 1 + most_likely_cat[ii,jj]*4
#                else:
#                    if for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.7:
#                        for_cat[ii,jj] = 1 + most_likely_cat[ii,jj]*4
#                    elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.6:
#                        for_cat[ii,jj] = 2 + most_likely_cat[ii,jj]*4
#                    elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.5:
#                        for_cat[ii,jj] = 3 + most_likely_cat[ii,jj]*4
#                    elif for_terciles[ii,jj,most_likely_cat[ii,jj]]>=0.4:
#                        for_cat[ii,jj] = 4 + most_likely_cat[ii,jj]*4
#
#    mascara = for_cat<1
#
#    for_mask = np.ma.masked_array(for_cat,mascara)
#
#    fig = plt.figure()
#    mapproj = bm.Basemap(projection='cyl', llcrnrlat=lats,
#    llcrnrlon= lonw, urcrnrlat= latn, urcrnrlon= lone)
#    #projection and map limits
#    mapproj.drawcoastlines()          # coast
#    mapproj.drawcountries()         #countries
#    lonproj, latproj = mapproj(dx, dy)      #poject grid
#
#    CS1 = mapproj.pcolor(lonproj, latproj, for_mask, cmap = cmap, vmin = 0.5, vmax = 12.5)
#    plt.title('DJF Precipitation Forecast IC Nov. ' + str(k) + ' - Uncalibrated')
#    ax1 = fig.add_axes([0.2, 0.05, 0.2, 0.03])
#    cmap1 = mpl.colors.ListedColormap(colores[0:4,:])
#    bounds = [0.5,1.5,2.5,3.5,4.5]
#    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
#    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap1, norm=norm, boundaries=bounds,
#            ticks=[1,2,3,4],spacing='uniform', orientation='horizontal')
#    cb1.set_ticklabels(['+70%','65%','55%','45%'])
#    cb1.ax.tick_params(labelsize = 7)
#    cb1.set_label('Lower')
#    ax2 = fig.add_axes([0.415, 0.05, 0.2, 0.03])
#    cmap2 = mpl.colors.ListedColormap(colores[4:8,:])
#    bounds = [4.5,5.5,6.5,7.5,8.5]
#    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
#    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap2, norm=norm, boundaries=bounds,
#            ticks=[5,6,7,8],spacing='uniform', orientation='horizontal')
#    cb2.set_ticklabels(['45%','55%','65%','+70%'])
#    cb2.ax.tick_params(labelsize = 7)
#    cb2.set_label('Normal')
#    ax3 = fig.add_axes([0.63, 0.05, 0.2, 0.03])
#    cmap3 = mpl.colors.ListedColormap(colores[8:,:])
#    bounds = [8.5,9.5,10.5,11.5,12.5]
#    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
#    cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap3, norm=norm, boundaries=bounds,
#            ticks=[9,10,11,12], spacing='uniform', orientation='horizontal')
#    cb3.set_ticklabels(['45%','55%','65%','+70%'])
#    cb3.ax.tick_params(labelsize = 7)
#    cb3.set_label('Upper')
#    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
#    plt.close()

#plot observed category
ruta = '/datos/osman/nmme_output/'
archivo = 'obs_prec_1983_DJF.npz'

data = np.load(ruta + archivo)
#clasifico los pronos

lat = data['lats_obs']
lon = data['lons_obs']
lats = np.min(lat)
latn = np.max(lat)
lonw = np.min(lon)
lone = np.max(lon)
[dx,dy] = np.meshgrid (lon,lat)
cmap = mpl.colors.ListedColormap(np.array([[217,95,14],[189,189,189],[44,162,95]])/256)

for k in np.arange(1982,2010):
    output = '/datos/osman/nmme_figuras/forecast/obs_cat_prec_DJF_ic_nov_' + str(k) + '.png'
    obs_cat = np.squeeze(data['cat_obs'][:,k-1982,:,:])
    nlats = obs_cat.shape[1]
    nlons = obs_cat.shape[2]
    #print(obs_cat[:,10,10])
    obs = np.zeros([nlats,nlons])
    obs[obs_cat[1,:,:]==1] = 1
    obs[obs_cat[2,:,:]==1] = 2
    obs[obs_cat[0,:,:]==1] = 0
    fig = plt.figure()
    mapproj = bm.Basemap(projection='cyl', llcrnrlat=lats,
    llcrnrlon= lonw, urcrnrlat= latn, urcrnrlon= lone)
    #projection and map limits
    mapproj.drawcoastlines()          # coast
    mapproj.drawcountries()         #countries
    lonproj, latproj = mapproj(dx, dy)      #poject grid
    CS1 = mapproj.pcolor(lonproj, latproj, obs, cmap = cmap, alpha = 0.6, vmin = -0.5, vmax = 2.5)
    plt.title('DJF Observed Category ' + str(k) )
    ax = fig.add_axes([0.42, 0.05, 0.2, 0.03])
    bounds = [-0.5,0.5,1.5,2.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, boundaries=bounds,
            ticks=[0,1,2],spacing='uniform', orientation='horizontal',alpha = 0.6)
    cb.set_ticklabels(['Lower','Middle','Upper'])
    cb.ax.tick_params(labelsize = 7)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.close()


