"""plot summary of verification indexes to compare combination techniques:
    - Mean RPSS, AUROc and BSS as as fuction of forecasted season for dif comb techniques
    - Reliability diagram for the main seasons
    
"""
import numpy as np
import calendar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#route to verification scores
ruta = '/datos/osman/nmme_output/verif_scores/'

ctech = ['wsereg', 'wpdf'] #combination technique
wtech = ['same', 'mean_cor', 'pdf_int'] #weighting technique
index = ['rpss', 'bss_above', 'bss_below', 'auroc_above', 'auroc_below']
mme_spread = 1.0
mod_spread = 1.0
#define dictionary with idexes for each comb and weight pair 
#extract latitudes
#compute mean indexes
#plot results
mean_index_array = np.zeros(7*5*12, dtype = {'names': index, 
    'formats': ['f8', 'f8', 'f8', 'f8', 'f8']})
mean_index_dict = []
jj = 0
for c in ctech:
    for w in wtech:

        for leadtime in np.arange(1, 6):
            for IC in np.arange(1, 13):
                
                seas = range(IC + leadtime, IC + leadtime + 3)
                sss = [i-12 if i>12 else i for i in seas]
                SSS = "".join(calendar.month_abbr[i][0] for i in sss)
                for ii in index:
                    if c == 'wsereg':
                        filename = ii + '_prec_mme_' + calendar.month_abbr[IC] + '_' + SSS +\
                                '_gp_01_p_' + '{:03}'.format(mod_spread) + '_' + w + '_' +\
                                c + '_p_' + '{:03}'.format(mme_spread) + '_hind.npz'
                    else:
                        filename = ii + '_prec_mme_' + calendar.month_abbr[IC] + '_' + SSS +\
                                '_gp_01_p_' + '{:03}'.format(mod_spread) + '_' + w +\
                                '_' + c + '_hind.npz' 
                    
                    data = np.load(ruta + filename)
                    #mean_index_dict.append({c + '_' + w + '_' + ii: np.nanmean(np.reshape(data [ii], 
                    #    data[ii].size))})
                    if (ii == 'bss_above') or (ii == 'bss_below'):
                        mean_index_dict.append(np.nanmean(np.reshape(1-data[ii][0,:,:]/(0.33*(1-0.33)),
                            data[ii][0,:,:].size)))
                    else:
                        mean_index_dict.append(np.nanmean(np.reshape(data[ii],data[ii].size)))

                          
                mean_index_array[jj] = tuple(np.array(mean_index_dict))
                jj = jj + 1
                mean_index_dict = []
w = 'same'
c = 'count'
#repeat for count
for leadtime in np.arange(1, 6):
    for IC in np.arange(1, 13):
        seas = range(IC + leadtime, IC + leadtime + 3)
        sss = [i-12 if i>12 else i for i in seas]
        SSS = "".join(calendar.month_abbr[i][0] for i in sss)
        for ii in index:
            filename = ii + '_prec_mme_' + calendar.month_abbr[IC] + '_' + SSS +\
                    '_gp_01_p_' + '{:03}'.format(mod_spread) + '_' + w +\
                    '_' + c + '_hind.npz' 
            
            data = np.load(ruta + filename)
            #mean_index_dict.append({c + '_' + w + '_' + ii: np.nanmean(np.reshape(data [ii], 
            #    data[ii].size))})
            if (ii == 'bss_above') or (ii == 'bss_below'):
                mean_index_dict.append(np.nanmean(np.reshape(1-data[ii][0,:,:]/(0.33*(1-0.33)),
                    data[ii][0,:,:].size)))
            else:
                mean_index_dict.append(np.nanmean(np.reshape(data[ii],data[ii].size)))

                  
        mean_index_array[jj] = tuple(np.array(mean_index_dict))
        jj = jj + 1
        mean_index_dict = []


#plot mean value of each index for each method and each lead 1 forecast
leyenda = []
for i in ctech:
    for j in wtech:
        leyenda.append(i + '-' + j)
leyenda.append('uncal')
#mean_index_list 2*3*7*12 elements

#plot mean index for each season and leatime
color = iter(cm.rainbow(np.linspace(0,1,7)))
route = '/datos/osman/nmme_figuras/synthesis_figures/'
for ii in index:
    aux = np.reshape(mean_index_array[ii],[7,5,12])
    for leadtime in np.arange(5):
        color = iter(cm.rainbow(np.linspace(0,1,7)))
        
        #me quedo solo con el leadtime que me interesa
        var = aux[:,leadtime,:]
        #plot
        plt.figure()
        for i in np.arange(np.shape(aux)[0]):
                c = next (color)
                plt.plot(var[i,:], c = c, label = leyenda[i])

        plt.xticks(np.arange(0,12), ('JFM','FMA','MAM','AMJ','MJJ','JJA','JAS',\
                'ASO','SON','OND','NDJ','DJF'),rotation = 20)
        plt.legend(bbox_to_anchor = (1.04,0.5), loc = "center left")
        plt.title('Mean '+ ii.upper() + ' - Leadtime '+ str(leadtime + 1) + ' month')
        plt.savefig(route + ii + '_leadtime_' + str(leadtime + 1) + '_month.png',
                dpi = 300, bbox_inches = 'tight', papertype = 'A4')
        plt.close()







            
