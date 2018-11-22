"""plot summary of verification INDEXes to compare combination techniques:
    - Reliability and ROC diagram for the main seasons
"""
import calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#ROUTE to verification scores
RUTA = '/datos/osman/nmme_output/verif_scores/'

CTECH = ['wsereg', 'wpdf'] #combination technique
WTECH = ['same', 'mean_cor', 'pdf_int'] #weighting technique
INDEX = ['hrr_above', 'farr_above', 'hrr_below', 'farr_below', 'hrrd_above',
         'hrrd_below']
DOMAIN = ['all', 'trop_SA', 'extratrop_SA']

mean_index_array = np.zeros([3, 7, 7, 12, 6, 10])
for d in np.arange(len(DOMAIN)):
    jj = 0
    for c in np.arange(len(CTECH)):
        for w in np.arange(len(WTECH)):
            for IM in np.arange(1, 13):
                seas = range(IM, IM + 3)
                sss = [i - 12 if i > 12 else i for i in seas]
                SSS = "".join(calendar.month_abbr[i][0] for i in sss)
                for leadtime in np.arange(1, 8):
                    IC = (IM - leadtime + 12) if (
                        (IM - leadtime) <= 0) else (IM - leadtime)
                    cc = CTECH[c]
                    ww = WTECH[w]
                    filename = 'rel_roc_' + DOMAIN[d] + '_prec_mme_' +\
                    calendar.month_abbr[IC] + '_' + SSS + '_gp_01_' + ww + '_'\
                    + cc + '_hind.npz'
                    data = np.load(RUTA + filename)
                    for ii in np.arange(len(INDEX)):
                        mean_index_array[d, jj, leadtime - 1, IC - 1,
                                         ii, :] = data[INDEX[ii]]
        jj = jj + 1

w = 'same'
c = 'count'
SSS_in = []
#repeat for count
for d in np.arange(len(DOMAIN)):
    for IM in np.arange(1, 13):
        seas = range(IM, IM + 3)
        sss = [i - 12 if i > 12 else i for i in seas]
        SSS = "".join(calendar.month_abbr[i][0] for i in sss)
        SSS_in.append(SSS)
        for leadtime in np.arange(1, 8):
            IC = (IM - leadtime + 12) if (
                (IM - leadtime) <= 0) else (IM - leadtime)
            filename = 'rel_roc_' + DOMAIN[d] + '_prec_mme_' +\
            calendar.month_abbr[IC] + '_' + SSS + '_gp_01_' + w + '_' + c +\
            '_hind.npz'
            data = np.load(RUTA + filename)
            for ii in np.arange(len(INDEX)):
                mean_index_array[d, -1, leadtime - 1, IC - 1,
                                 ii, :] = data[INDEX[ii]]

leyenda = []
for i in CTECH:
    for j in WTECH:
        leyenda.append(i + '-' + j)
leyenda.append('uncal')

#plot mean INDEX for each season and leatime
ROUTE = '/datos/osman/nmme_figuras/synthesis_figures/'
#reliability:hrrd
for ii in [4, 5]:
    for d in np.arange(3): #loop over DOMAINs        
        for leadtime in np.arange(7):
            for season in np.arange(12):
                color = iter(cm.rainbow(np.linspace(0, 1, 7)))
                #me quedo solo con el leadtime y estacion que me interesa
                var = mean_index_array[d, :, leadtime, season, ii, :]
                #plot
                plt.figure()
                for i in np.arange(np.shape(var)[0]):
                    c = next(color)
                    if i == 6:
                        plt.plot(np.arange(0.05, 1.05, 0.1), var[i, :], c=c,
                                 marker='o', label=leyenda[i], lw=2, ms=0.8)
                    else:
                        plt.plot(np.arange(0.05, 1.05, 0.1),var[i,:], c=c,
                                 marker='o', label=leyenda[i], lw=1.4, ms=0.8)
                plt.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12),
                         color='k')
                plt.axis([0, 1, 0, 1])
                plt.xlabel('Mean Forecast Probability')
                plt.ylabel('Observed Relative Frequency')
                plt.legend(bbox_to_anchor=(0.04, 0.85), loc="upper left")
                plt.title('Reliability '+ INDEX[ii][5:].upper() + '- ' +
                          SSS_in[season] + ' - Leadtime '+ str(leadtime + 1) +
                          ' month - ' + DOMAIN[d])
                plt.savefig(ROUTE + 'rel_' + INDEX[ii][5:] + '_' + DOMAIN[d] +
                            '_' + SSS_in[season] + '_leadtime_' + str(leadtime
                                                                      + 1) +
                            '_month.png', dpi=600, bbox_inches='tight',
                            papertype='A4')
                plt.close()
#roc
for ii in np.arange(2):
    for d in np.arange(3): #loop over DOMAINs
        for leadtime in np.arange(7):
            for season in np.arange(12):
                color = iter(cm.rainbow(np.linspace(0, 1, 7)))
                #me quedo solo con el leadtime y estacion que me interesa
                if ii == 0:
                    var1 = mean_index_array[d, :, leadtime, season, 0, :]
                    var2 = mean_index_array[d, :, leadtime, season, 1, :]
                else:
                    var1 = mean_index_array[d, :, leadtime, season, 2, :]
                    var2 = mean_index_array[d, :, leadtime, season, 3, :]
                #plot
                plt.figure()
                for i in np.arange(np.shape(var)[0]):
                    c = next(color)
                    plt.plot(var2[i, :] + 0.01, var1[i, :] + 0.01, c=c,
                             marker='o', label=leyenda[i], lw=1.4, ms=0.8)
                plt.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12),
                         color='k')
                plt.axis([0, 1, 0, 1])
                plt.xlabel('False Alarm Rate')
                plt.ylabel('Hit Rate')
                plt.legend(bbox_to_anchor=(0.9, 0.05), loc="lower right")
                plt.title('Roc Diagram '+ INDEX[ii * 2][4:].upper() + '- ' +
                          SSS_in[season] + ' - Leadtime ' + str(leadtime + 1) +
                          ' month - ' + DOMAIN[d])
                plt.savefig(ROUTE + 'roc_' + INDEX[ii * 2][4:] + '_' +
                            DOMAIN[d] + '_' + SSS_in[season] + '_leadtime_' +
                            str(leadtime + 1) + '_month.png', dpi=600,
                            bbox_inches='tight', papertype='A4')
                plt.close()
