"""plot summary of verification indexes to compare combination techniques:
    - Mean RPSS, AUROc and BSS as as fuction of forecasted season for dif comb techniques
    - Reliability diagram for the main seasons
"""
import calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#route to verification scores
ruta = '/datos/osman/nmme_output/verif_scores/'

ctech = ['wsereg', 'wpdf'] #combination technique
wtech = ['same', 'mean_cor', 'pdf_int'] #weighting technique
index = ['rpss', 'bss_above', 'bss_below', 'auroc_above', 'auroc_below']

mean_index_array = np.zeros(7 * 7 * 12,
                            dtype={'names': index,
                                   'formats': ['f8', 'f8', 'f8', 'f8', 'f8']})
mean_index_array_trop_SA = np.zeros(7 * 7 * 12,
                                    dtype={'names': index,
                                           'formats': ['f8', 'f8', 'f8', 'f8',
                                                       'f8']})
mean_index_array_extratrop_SA = np.zeros(7 * 7 * 12,
                                         dtype={'names': index,
                                                'formats': ['f8', 'f8', 'f8',
                                                            'f8', 'f8']})
mean_index_dict = []
mean_index_dict_trop_SA = []
mean_index_dict_extratrop_SA = []
#SA tropical: north of 20S 85W - 30W
latn_trop = 15
lats_trop = -20
lonw_trop = 275
lone_trop = 330
#SA extratropical 20S-55S 292-308
latn_extrop = -20
lats_extrop = -55
lonw_extrop = 292
lone_extrop = 308

jj = 0
for c in ctech:
    for w in wtech:
        for IM in np.arange(1, 13):
            seas = range(IM, IM + 3)
            sss = [i - 12 if i > 12 else i for i in seas]
            SSS = "".join(calendar.month_abbr[i][0] for i in sss)
            for leadtime in np.arange(1, 8):
                IC = (IM - leadtime + 12) if (IM - leadtime) <= 0\
                else (IM - leadtime)
                for ii in index:
                    if c == 'wsereg':
                        filename = ii + '_prec_mme_' + calendar.month_abbr[IC] + '_' + SSS +\
                                '_gp_01_' + w + '_' + c + '_hind.npz'
                    else:
                        filename = ii + '_prec_mme_' + calendar.month_abbr[IC] + '_' + SSS +\
                                '_gp_01_' + w + '_' + c + '_hind.npz'

                    data = np.load(ruta + filename)
                    lat = data['lat']
                    lon = data['lon']

                    lati_trop = np.argmin(abs(lat - lats_trop))
                    latf_trop = np.argmin(abs(lat - latn_trop)) + 1
                    loni_trop = np.argmin(abs(lon - lonw_trop))
                    lonf_trop = np.argmin(abs(lon - lone_trop)) + 1
                    lati_extrop = np.argmin(abs(lat - lats_extrop))
                    latf_extrop = np.argmin(abs(lat - latn_extrop)) + 1
                    loni_extrop = np.argmin(abs(lon - lonw_extrop))
                    lonf_extrop = np.argmin(abs(lon - lone_extrop)) + 1

                    if (ii == 'bss_above') or (ii == 'bss_below'):
                        mean_index_dict.append(np.nanmean(
                            np.reshape(1 - data[ii][0, :, :] / (0.33 * (1 -
                                                                        0.33)),
                                       data[ii][0, :, :].size)))
                        mean_index_dict_trop_SA.append(np.nanmean(
                            np.reshape(1 - data[ii][0, lati_trop:latf_trop,\
                                                    loni_trop:lonf_trop] /
                                       (0.33 * (1 - 0.33)),
                                       (data[ii][0, lati_trop:latf_trop,\
                                                 loni_trop:lonf_trop]).size)))
                        mean_index_dict_extratrop_SA.append(np.nanmean(
                            np.reshape(1 -  data[ii][0, \
                                                     lati_extrop:latf_extrop,\
                                                     loni_extrop:lonf_extrop]
                                       /(0.33 * (1 - 0.33)),
                                       (data[ii][0, lati_extrop:latf_extrop,\
                                                 loni_extrop:lonf_extrop]
                                       ).size)))
                    else:
                        mean_index_dict.append(np.nanmean(
                            np.reshape(data[ii], data[ii].size)))
                        mean_index_dict_trop_SA.append(np.nanmean(
                            np.reshape(data[ii][lati_trop:latf_trop,
                                                loni_trop:lonf_trop],
                                       (data[ii][lati_trop:latf_trop,
                                                 loni_trop:lonf_trop]
                                       ).size)))
                        mean_index_dict_extratrop_SA.append(np.nanmean(
                            np.reshape(data[ii][lati_extrop:latf_extrop,
                                                loni_extrop:lonf_extrop],
                                       (data[ii][lati_extrop:latf_extrop,
                                                 loni_extrop:lonf_extrop]
                                       ).size)))


                mean_index_array[jj] = tuple(np.array(mean_index_dict))
                mean_index_array_trop_SA[jj] = tuple(np.array(mean_index_dict_trop_SA))
                mean_index_array_extratrop_SA[jj] = tuple(np.array(mean_index_dict_extratrop_SA))

                jj = jj + 1
                mean_index_dict = []
                mean_index_dict_trop_SA = []
                mean_index_dict_extratrop_SA = []
w = 'same'
c = 'count'
#repeat for count
for IM in np.arange(1, 13):
    seas = range(IM, IM + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    for leadtime in np.arange(1, 8):
        IC = (IM - leadtime + 12) if (IM - leadtime) <= 0 else (IM - leadtime)
        for ii in index:
            filename = ii + '_prec_mme_' + calendar.month_abbr[IC] + '_' + SSS +\
                    '_gp_01_' + w + '_' + c + '_hind.npz'
            data = np.load(ruta + filename)
            if (ii == 'bss_above') or (ii == 'bss_below'):
                mean_index_dict.append(np.nanmean(
                    np.reshape(1 - data[ii][0, :, :] / (0.33 * (1 - 0.33)),
                               data[ii][0, :, :].size)))
                mean_index_dict_trop_SA.append(np.nanmean(
                    np.reshape(1 - data[ii][0, lati_trop:latf_trop, loni_trop:
                                            lonf_trop] / (0.33 * (1 - 0.33)),
                               data[ii][0, lati_trop:latf_trop, loni_trop:\
                                          lonf_trop].size)))
                mean_index_dict_extratrop_SA.append(np.nanmean(
                    np.reshape(1 - data[ii][0, lati_extrop:latf_extrop,
                                            loni_extrop:lonf_extrop] /
                               (0.33 * (1 - 0.33)),
                               data[ii][0, lati_extrop:latf_extrop,
                                        loni_extrop:lonf_extrop].size)))
            else:
                mean_index_dict.append(np.nanmean(
                    np.reshape(data[ii], data[ii].size)))
                mean_index_dict_trop_SA.append(np.nanmean(
                    np.reshape(data[ii][lati_trop:latf_trop,
                                        loni_trop:lonf_trop],
                               data[ii][lati_trop:latf_trop,
                                        loni_trop:lonf_trop].size)))
                mean_index_dict_extratrop_SA.append(np.nanmean(
                    np.reshape(data[ii][lati_extrop:latf_extrop,
                                        loni_extrop:lonf_extrop],
                               data[ii][lati_extrop:latf_extrop,
                                        loni_extrop:lonf_extrop].size)))

        mean_index_array[jj] = tuple(np.array(mean_index_dict))
        mean_index_array_trop_SA[jj] = tuple(np.array(mean_index_dict_trop_SA))
        mean_index_array_extratrop_SA[jj] = tuple(np.array(mean_index_dict_extratrop_SA))
        jj = jj + 1
        mean_index_dict = []
        mean_index_dict_trop_SA = []
        mean_index_dict_extratrop_SA = []

#plot mean value of each index for each method and each lead 1 forecast
leyenda = []
for i in ctech:
    for j in wtech:
        leyenda.append(i + '-' + j)
leyenda.append('uncal')

#plot mean index for each season and leatime

route = '/datos/osman/nmme_figuras/synthesis_figures/mean_score/'
for ii in index:
    aux = np.reshape(mean_index_array[ii], [7, 12, 7])
    for leadtime in np.arange(7):
        color = iter(cm.rainbow(np.linspace(0, 1, 7)))
        #me quedo solo con el leadtime que me interesa
        var = aux[:, :, leadtime]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
            c = next(color)
            plt.plot(var[i, :], c=c, label=leyenda[i])
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.axis([-1, 12, 0, 0.6])
        else:
            plt.axis([-1, 12, -0.2, 0.3])

        plt.xticks(np.arange(0, 12), ('JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                      'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ',
                                      'DJF'), rotation=20)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.title('Mean ' + ii.upper() + ' - Leadtime ' + str(leadtime + 1) +
                  ' month - ALL')
        plt.savefig(route + ii + '_leadtime_' + str(leadtime + 1) +
                    '_month_all.png', dpi=300, bbox_inches='tight',
                    papertype='A4')
        plt.close()
for ii in index:
    aux = np.reshape(mean_index_array_trop_SA[ii], [7, 12, 7])
    for leadtime in np.arange(7):
        color = iter(cm.rainbow(np.linspace(0, 1, 7)))
        #me quedo solo con el leadtime que me interesa
        var = aux[:, :, leadtime]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
            c = next(color)
            plt.plot(var[i, :], c=c, label=leyenda[i])
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.axis([-1, 12, 0, 0.6])
        else:
            plt.axis([-1, 12, -0.2, 0.3])
        plt.xlabel('Target Season')
        plt.xticks(np.arange(0, 12), ('JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                      'JJA', 'JAS', 'ASO', 'SON', 'OND',
                                      'NDJ', 'DJF'), rotation=20)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.title('Mean '+ ii.upper() + ' - Leadtime '+ str(leadtime + 1) +
                  ' month - Trop SA')
        plt.savefig(route + ii + '_leadtime_' + str(leadtime + 1) +
                    '_month_trop_SA.png', dpi=300, bbox_inches='tight',
                    papertype='A4')
        plt.close()
for ii in index:
    aux = np.reshape(mean_index_array_extratrop_SA[ii], [7, 12, 7])
    for leadtime in np.arange(7):
        color = iter(cm.rainbow(np.linspace(0, 1, 7))) 
        #me quedo solo con el leadtime que me interesa
        var = aux[:, :, leadtime]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
            c = next(color)
            plt.plot(var[i, :], c=c, label=leyenda[i])
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.axis([-1, 12, 0, 0.6])
        else:
            plt.axis([-1, 12, -0.2, 0.3])
        plt.xlabel('Target Season')
        plt.xticks(np.arange(0, 12), ('JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS',\
                'ASO', 'SON', 'OND', 'NDJ', 'DJF'), rotation=20)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.title('Mean '+ ii.upper() + ' - Leadtime '+ str(leadtime + 1) +
                  ' month - Extratrop SA')
        plt.savefig(route + ii + '_leadtime_' + str(leadtime + 1) +
                    '_month_extrop_SA.png', dpi=300, bbox_inches='tight',
                    papertype='A4')
        plt.close()

#for each cal and comb method plot mean score for each season vs leadtime
route = '/datos/osman/nmme_figuras/synthesis_figures/mean_score_box/'
for ii in index:
    aux = np.reshape(mean_index_array[ii], [7, 12, 7])
    for i in np.arange(7):
        cmap = plt.cm.get_cmap('PuOr', 12)
        plt.figure()
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.pcolor(np.transpose(aux[i, :, :]), edgecolors='k', linewidth=2,
                       cmap=cmap, vmin=-0.6, vmax=0.6)
        else:
            plt.pcolor(np.transpose(aux[i, :, :]), edgecolors='k', linewidth=2,
                       cmap=cmap, vmin=-0.3, vmax=0.3)

        plt.xlabel('Target Season')
        plt.ylabel('Leadtime')
        plt.xticks(np.arange(0.5, 12.5, 1), ('JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                             'JJA', 'JAS', 'ASO', 'SON', 'OND',
                                             'NDJ', 'DJF'), rotation=20)
        plt.yticks(np.arange(0.5, 7.5, 1), ('1', '2', '3', '4', '5', '6', '7'))
        plt.colorbar()
        plt.title('Mean '+ ii.upper() + ' -  '+ leyenda[i] +' - ALL')
        plt.savefig(route + ii + '_all_' + leyenda[i] + '.png', dpi=300,
                    bbox_inches='tight', papertype='A4')
        plt.close()
for ii in index:
    aux = np.reshape(mean_index_array_trop_SA[ii], [7, 12, 7])
    for i in np.arange(7):
        cmap = plt.cm.get_cmap('PuOr', 12)
        #bound = np
        plt.figure()
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.pcolor(np.transpose(aux[i, :, :]), edgecolors='k', linewidth=2,
                       cmap=cmap, vmin=-0.6, vmax=0.6)
        else:
            plt.pcolor(np.transpose(aux[i, :, :]), edgecolors='k', linewidth=2,
                       cmap=cmap, vmin=-0.3, vmax=0.3)

        plt.xlabel('Target Season')
        plt.ylabel('Leadtime')
        plt.xticks(np.arange(0.5, 12.5, 1), ('JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                             'JJA', 'JAS', 'ASO', 'SON', 'OND',
                                             'NDJ', 'DJF'), rotation=20)
        plt.yticks(np.arange(0.5, 7.5, 1), ('1', '2', '3', '4', '5', '6', '7'))
        plt.colorbar()
        plt.title('Mean '+ ii.upper() + ' -  '+ leyenda[i] + ' - Tropical SA')
        plt.savefig(route + ii + '_trop_SA_' + leyenda[i] + '.png', dpi=300,
                    bbox_inches='tight', papertype='A4')
        plt.close()

for ii in index:
    aux = np.reshape(mean_index_array_extratrop_SA[ii], [7, 12, 7])
    for i in np.arange(7):
        cmap = plt.cm.get_cmap('PuOr', 12)
        #bound = np
        plt.figure()
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.pcolor(np.transpose(aux[i, :, :]), edgecolors='k', linewidth=2,
                       cmap=cmap, vmin=-0.6, vmax=0.6)
        else:
            plt.pcolor(np.transpose(aux[i, :, :]), edgecolors='k', linewidth=2,
                       cmap=cmap, vmin=-0.3, vmax=0.3)

        plt.xlabel('Target Season')
        plt.ylabel('Leadtime')
        plt.xticks(np.arange(0.5, 12.5, 1), ('JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                             'JJA', 'JAS', 'ASO', 'SON', 'OND',
                                             'NDJ', 'DJF'), rotation=20)
        plt.yticks(np.arange(0.5, 7.5, 1), ('1', '2', '3', '4', '5', '6', '7'))
        plt.colorbar()
        plt.title('Mean '+ ii.upper() + ' - '+ leyenda[i] + ' - Extratropical'
                  + 'SA')
        plt.savefig(route + ii + '_extratrop_SA_' + leyenda[i] + '.png',
                    dpi=300, bbox_inches='tight', papertype='A4')
        plt.close()

#plot mean score vs lead for all cal-comb method
route = '/datos/osman/nmme_figuras/synthesis_figures/scores_vs_leadtime/'
seas = ['JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND',
        'NDJ', 'DJF']
for ii in index:
    aux = np.reshape(mean_index_array[ii], [7, 12, 7])
    for season in np.arange(12):
        color = iter(cm.rainbow(np.linspace(0, 1, 7)))
        #me quedo solo con el leadtime que me interesa
        var = aux[:, season, :]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
            c = next(color)
            plt.plot(var[i, :], c=c, label=leyenda[i])
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.axis([-1, 7, 0, 0.6])
        else:
            plt.axis([-1, 7, -0.1, 0.25])

        plt.xticks(np.arange(0, 7), ('1', '2', '3', '4', '5', '6', '7'))
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.title('Mean ' + ii.upper() + ' - Target: '+ seas[season])
        plt.savefig(route + ii + '_all_' + seas[season] + '.png',
                    dpi=300, bbox_inches='tight', papertype='A4')
        plt.close()
for ii in index:
    aux = np.reshape(mean_index_array_trop_SA[ii], [7, 12, 7])
    for season in np.arange(12):
        color = iter(cm.rainbow(np.linspace(0, 1, 7)))
        #me quedo solo con el leadtime que me interesa
        var = aux[:, season, :]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
            c = next(color)
            plt.plot(var[i, :], c=c, label=leyenda[i])
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.axis([-1, 7, 0, 0.6])
        else:
            plt.axis([-1, 7, -0.1, 0.25])
        plt.xticks(np.arange(0, 7), ('1', '2', '3', '4', '5', '6', '7'))
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.title('Mean ' + ii.upper() + ' - Target: '+ seas[season] +
                  ' - Trop SA')
        plt.savefig(route + ii + '_trop_SA_' + seas[season] + '.png',
                    dpi=300, bbox_inches='tight', papertype='A4')
        plt.close()
for ii in index:
    aux = np.reshape(mean_index_array_extratrop_SA[ii], [7, 12, 7])
    for season in np.arange(12):
        color = iter(cm.rainbow(np.linspace(0, 1, 7)))
        #me quedo solo con el leadtime que me interesa
        var = aux[:, season, :]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
            c = next(color)
            plt.plot(var[i, :], c=c, label=leyenda[i])
        if (ii == 'auroc_above') or (ii == 'auroc_below'):
            plt.axis([-1, 7, 0, 0.6])
        else:
            plt.axis([-1, 7, -0.1, 0.25])
        plt.xticks(np.arange(0, 7), ('1', '2', '3', '4', '5', '6', '7'))
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.title('Mean ' + ii.upper() + ' - Target: '+ seas[season] +
                  ' - Extratrop SA')
        plt.savefig(route + ii + '_extratrop_SA_' + seas[season] + '.png',
                    dpi=300, bbox_inches='tight', papertype='A4')
        plt.close()
