import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()

#funciones para verificar pronosticos

def hit_false_alarm(Occurrence,Pforecast,prob_bin_vector,lats):

    # computes obs and non-obs rate for each prob_bin_interval
    #it follows WMO recommendation and weights each gridpoint with its latitude
    
    # INPUT
    # Pforecast: forecasted probability
    #Occurrence: 1/0 vector for yes/no ocurrence of the event
    #prob_bin_vector: lower and upper limit of each bin (from o to 1)
    #lats: matrix of latitud of each gridpoint in degrees
    # OUTPUT
    # O: array of nbins-1 lenght with the weighted sum of the observations for each bin
    # NO: same as O but for Non ocurrence

    nbins = np.shape(prob_bin_vector)[0]
    O = np.empty([nbins-1])
    NO = np.empty([nbins-1])

    for i in np.arange(nbins-1):

            cases = np.logical_and(Pforecast[:]>prob_bin_vector[i],
                        Pforecast[:]<=prob_bin_vector[i+1])
            O[i] = np.nansum(Occurrence[cases]*np.cos(np.deg2rad(lats[cases])))
            NO[i] = np.nansum(np.logical_not(Occurrence[cases])*
                        np.cos(np.deg2rad(lats[cases])))
    return O, NO

    
def plot_scores(lat, lon, var, titulo, output):
        #funcion para graficar scores (ergo barra entre -1 y 1)
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
    mapproj.drawcountries(linewidth = 0.2)         #countries
    lonproj, latproj = mapproj(dx, dy)      #poject grid
    # set desired contour levels.
    clevs = np.linspace(-1.1,1.1 , 13)
    barra = plt.cm.get_cmap('coolwarm',11) #colorbar
    CS1 = mapproj.pcolor(lonproj, latproj, var, cmap = barra, vmin = -1.1, vmax = 1.1)
    cbar = plt.colorbar(CS1, ticks = np.linspace(-0.9,0.9,10))
    cbar.ax.tick_params(labelsize = 8)
    plt.title(titulo)
    plt.savefig(output, dpi=300, bbox_inches='tight', papertype='A4')
    plt.close()

    return

def BS_decomposition(Pforecast, Occurrence, prob_bin_vector):

    #[BS]=BS_decomposition(Pforecast,Occurrence, prob_bin_vector)
    #
    #Brier score decomposition, implemented by Steven Weijs, Water Resources Management, TU Delft
    #
    #calculates the 5 components of the brier score based on a series of
    #probabilistic forecasts and cor.pngonding occurrences of binary events.
    #Fast version, no input checking. 
    #Input:
        #  Pforecast: vector of forecasted probabilities of the event occuring. (between 0 and 1)
        #  Occurence: vector of zeros and ones for whether the event occurred 
        #  prob_bin_vector: the forecast probabilties can be put in several bins
        #       e.g. [0 0.33 0.66 1] for terciles
        #       in the case of binning, WBV and WBC components matter (see ref below)
    #Output: 
    #  struct BS with fields : 
    # .direct       Total brierscore
    # .unc          Uncertainty component
    # .res          Resolution
    # .rel          Reliability
    # .wbv          Within Bin variance
    # .wbc          Within bin covariance
    #
    #Total brierscore,and its components: reliability, resolution, uncertainty 
    #  see Murphy 1973
    #resolution and reliability are stratified on all issued probabilities if no
    #binning vector is specified this is the original murphy implementation. 
    #
    #if binning is used, two additional components are calculated:
    # WBV= within bin variance
    # WBC= within bin covariance
    # these can be used to calculate the generalized resolution component:
    # GRES = RES-WBV+WBC     (generalized resolution)
    # BS=REL-RES+UNC-WBV-WBC (brier score in five components)
    # see Stephenson, Coelho, Jolifffe 2007 for discussion about WBV WBC components

    if len(locals())>2:#probability bins specified, use them
        
        binning = True
        prob_bins = np.transpose(np.array([(prob_bin_vector[0:-1]), (prob_bin_vector[1:])]))
        prob_bins[0,0] =-1  #trick to include 0 in first bin

        
    else: #probability bins not specified, use all issued probs 

        binning = False
        prob_bins = np.unique(Pforecast)

    
    #Total
    BS ={'direct': np.nanmean(np.power(Pforecast-Occurrence,2))}
    #number of forecasts issued and number of bins to stratify
    N = np.max(np.shape(Pforecast))
    Nk = np.shape(prob_bins)[0]
    
    if N==Nk: #all forecasts are unique
        shortcut = True
    else:
        shortcut = False

    #initialize vectors

    res = np.zeros([Nk])
    rel =  np.zeros([Nk])
    wbv =  np.zeros([Nk])
    wbc =  np.zeros([Nk])
    #climatic probability of occurrence
    pclim = np.nanmean(Occurrence)
    #uncertainty component
    BS['unc'] = (1-pclim)*pclim
    
    if shortcut:
        BS_1 =np.empty([6])
        BS_1[0] = BS['direct']
        BS_1[1] = BS['unc']
        BS_1[2] = BS['unc']
        BS_1[3] = BS['direct']
        BS_1[4] = 0 
        BS_1[5] = 0

        return BS_1

    #loop for all probability bins (or issued probs if no binning)
    for k in np.arange(Nk):
        #idxK= indexes that belong to bin K
        if binning: 
            #find indexes of forecasts within bin
            idxK = np.logical_and(Pforecast>prob_bins[k,0],Pforecast<=prob_bins[k,1])

        else:
            #find indexes of forecasts for a bin consisting of one prob only
            idxK = Pforecast==prob_bins[k]

        #values per bin to calculate components
        fk = Pforecast[idxK]   #vector of forecast probs within bin k
        occk = Occurrence[idxK] #vector of occurences within bin k
        avfk = np.nanmean(fk)     #average forecast prob within bin k
        avocck = np.nanmean(occk)  #average occurence (= probability of occurence) within bin k 
        nk = np.shape(fk)[0]     #number of values in bin k
        #reliability component (needs to be summed over bins later) 
        rel[k] = nk*(np.power(avfk-avocck,2))
        #resolution component (needs to be summed over bins later)
        res[k]=nk*(np.power(avocck-pclim,2))
        # xxx calculate wbv and wbc here
        if binning:

            wbv[k]=np.var(fk)*nk
            tmp=np.cov(fk,occk)*nk
            wbc[k]=tmp[1,0]
            #line to avoid problems with empty bins
      
            if nk==0:

                wbv[k] = 0
                wbc[k] = 0
                rel[k] = 0
                res[k] = 0

    #calculate total components by summing over all bins and dividing by nr obs
    BS['rel'] = np.nansum(rel)/N
    BS['res'] = np.nansum(res)/N

    if binning:
       
        BS['wbv'] = np.nansum(wbv)/N
        BS['wbc'] = np.nansum(wbc)/N

    else:

        BS['wbv'] = 0
        BS['wbc'] = 0

    BS_1 =np.empty([6])
    BS_1[0] = BS['direct']
    BS_1[1] = BS['unc']
    BS_1[2] = BS['res']
    BS_1[3] = BS['rel']
    BS_1[4] = BS['wbv']
    BS_1[5] = BS['wbc']

    return BS_1

def RPSS (Pforecast, Occurrence):

    #compute the Ranked Proabilistic skill score for terciles taken climatology 
    #as ref forecast

    #input:
    #PForecast: cumulate probabilistic forecast format ([[pbelow,pbelow+normal],years,gridpoints])
    #observed category (0 not observed 1 observed) [[B,N,A],years,gridpoints]]

    #output
    #RPSS [gridpoints]

    #compute RPS
    rps = np.nanmean((np.power(Pforecast[0,:,:,:]- Occurrence[0,:,:,:],2) +
              np.power(Pforecast[1,:,:,:]-np.sum(Occurrence[0:2,:,:,:], axis = 0),2)), 
              axis = 0)

    #compute RPS for ref 
    rps_ref = np.nanmean((np.power(0.33-Occurrence[0,:,:,:],2) +
                np.power(0.66-np.nansum(Occurrence[0:2,:,:,:],axis = 0),2)), axis = 0)

    rpss=1-rps/rps_ref
    return rpss

def auroc(Pforecast,Occurrence,lats,bins):
    
    #compute the Area under ROC diagram for a given event using bins as prob interval
    #input:
    #Pforecast: forecast probability of a given event
    #Occurrence: 1/0 vector for yes/no ocurrence of event
    #bins: probability intervals
    #lats: vector of latitudes
    #output: AUROC score using climatology as reference forecas (Area=0.5)
    
    def PolyArea(x,y):

        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    #Occurrence [years lat lon]

    [nyears,nlats,nlons] = np.shape(Occurrence)

   #llamo a la funcion que calculas las observaciones y no obsersavciones de cada
    #categoria para calcular luego el roc y rel diagram

    Obs = np.empty([len(bins)-1, nlats, nlons])
    NObs = np.empty([len(bins)-1, nlats, nlons])

    for i in np.arange(nlats):
        for j in np.arange(nlons):

            [Obs[:,i,j],NObs[:,i,j]] = hit_false_alarm(Occurrence[:,i,j],Pforecast[:,i,j],
                    bins,np.tile(lats[i],nyears))

    #ahora calculo el roc 
    #%saco el hit rate y false alarms rate para el roc score

    hrr = np.empty([len(bins-1), nlats,nlons])
    farr = np.empty([len(bins-1), nlats,nlons])

    for i in np.arange(len(bins-1)):

        #roc
        hrr[i,:,:] = np.sum(Obs[i:,:,:],axis=0)/np.sum(Obs,axis = 0)
        farr[i,:,:] = np.sum(NObs[i:,:,:],axis=0)/np.sum(NObs,axis = 0)

    auroc = np.empty([nlats, nlons])

    for i in np.arange(nlats):
        for j in np.arange(nlons):
            
            auroc[i,j] = 2*(PolyArea(np.concatenate([np.flip(farr[:,i,j],0),
                np.array([ 1, 0])]),np.concatenate([np.flip(hrr[:,i,j],0),
                    np.array( [0, 0])]))-0.5)
            
    return auroc

def rel_roc(Pforecast,Occurrence,lats,bins,ruta,output_name):
    
    #compute the reliability and ro diagram for terciles forecast using 
    #bins as prob interval
    #input:
    #Pforecast: cumulative probability forecast terciles pbelow, pbelow+pnormal
    #Occurrence: 1/0 vector for yes/no ocurrence of each tercile
    #bins: probability intervals
    #lats: vector of latitudes
    #output_name: used in the names of the figures
    #output: reliability and roc diagram for above normal and below normal events
    
    def PolyArea(x,y):

        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    [n,nyears,nlats,nlons] = np.shape(Occurrence)

    #replico el vector de latitudes por la cantidad de longitudes para usar
    #en el codigo
    latsrep = np.reshape(np.rollaxis(np.tile(lats,[nyears, nlons, 1]),2,1),nyears*nlats*nlons)
    Occurrence_below = np.reshape(Occurrence[0,:,:,:],nyears*nlats*nlons)
    Pforecast_below = np.reshape(Pforecast[0,:,:,:],nyears*nlats*nlons)
    Occurrence_above = np.reshape(Occurrence[2,:,:,:],nyears*nlats*nlons)
    Pforecast_above = 1-np.reshape(Pforecast[1,:,:,:],nyears*nlats*nlons)

    #llamo a la funcion que calculas las observaciones y no obsersavciones de cada
    [O_above,NO_above] = hit_false_alarm(Occurrence_above,Pforecast_above,bins,latsrep)
    [O_below,NO_below] = hit_false_alarm(Occurrence_below,Pforecast_below,bins,latsrep)

    #compute roc and reliability diagrams

    #hit rate and false alarm rate
    hrr_above = np.empty([len(bins)-1])
    farr_above = np.empty([len(bins)-1])
    hrr_below = np.empty([len(bins)-1])
    farr_below = np.empty([len(bins)-1])

    for i in np.arange(0,len(bins)-1):

        #roc

        hrr_above[i] = np.sum(O_above[i:])/np.sum(O_above)
        farr_above[i] = np.sum(NO_above[i:])/np.sum(NO_above)
        hrr_below[i] = np.sum(O_below[i:])/np.sum(O_below)
        farr_below[i] = np.sum(NO_below[i:])/np.sum(NO_below)

    #reliability

    hrrd_above = O_above/(O_above + NO_above)
    hrrd_below = O_below/(O_below + NO_below)

    #histograma de frecuencia
    frd_above = (O_above+NO_above)/np.shape(Occurrence_above)[0]
    frd_below = (O_below+NO_below)/np.shape(Occurrence_below)[0]
    roc_above = 2*(PolyArea(np.concatenate([np.flip(farr_above,0), np.array([1, 0])]),
                np.concatenate([np.flip(hrr_above,0),np.array([ 0, 0])]))-0.5)
    roc_below = 2*(PolyArea(np.concatenate([np.flip(farr_below,0), np.array([1, 0])]),
                np.concatenate([np.flip(hrr_below,0),np.array([ 0, 0])]))-0.5)

    ley1 = 'above: '+ '{:.3f}'.format(roc_above)
    ley2 = 'below: '+ '{:.3f}'.format(roc_below)

    #grafico el ROC

    plt.figure(1)
    plt.plot(farr_above,hrr_above,color = 'b',marker = 'o')
    plt.plot(farr_below,hrr_below,color = 'r',marker = 'o')
    for i in np.arange(1,np.shape(farr_above)[0]):
        plt.text(farr_above[i]+0.01,hrr_above[i]+0.01,
                str(bins[i]),color = 'k',fontsize=7)
        plt.text(farr_below[i]+0.01,hrr_below[i]+0.01,
                str(bins[i]),color='k',fontsize=7)

    plt.plot(np.linspace(0,1.1,12),np.linspace(0,1.1,12),color = 'k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.legend([ley1,ley2],loc = 'lower right')
    plt.title('Roc Diagram')
    plt.savefig(ruta+'roc/roc_'+output_name+'.png',dpi = 300, bbox_inches = 'tight', 
            papertype = 'A4')
    plt.close()

    #grafico el diagram de confiabilidad

    hf1 = plt.figure(2)
    plt.plot(np.arange(0.05,1.05,0.1),hrrd_above,color = 'b',marker = 'o')
    plt.plot(np.arange(0.05,1.05,0.1),hrrd_below,color = 'r',marker = 'o')
    plt.plot(np.linspace(0,1.1,12),np.linspace(0,1.1,12), color = 'k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Mean Forecast Probability')
    plt.ylabel( 'Observed Relative Frequency')
    plt.title('Reliability Diagram')
    # normalized units are used for position
    #%histograma above
    ax1 = plt.axes([0.15, 0.68, 0.15, 0.15])
    plt.bar(np.arange(0.05,1.05,0.1),frd_above,0.1)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.xaxis.label.set_size(6)
    ax1.yaxis.label.set_size(6)
    plt.axis([0, 1, 0, 1])
    plt.title('Above',fontsize = 10)
    #histograma below
    ax2 = plt.axes([0.73, 0.18, 0.15, 0.15])
    plt.bar(np.arange(0.05,1.05,0.1),frd_below,0.1, color = 'r')
    #set(gca,'xticklabel','','YAxisLocation', 'left')
    ax2.xaxis.label.set_size(6)
    ax2.yaxis.label.set_size(6)
    plt.axis([0, 1, 0, 1])
    plt.title('Below',fontsize = 10)
    plt.savefig(ruta+'rel/reliability_diagram_'+output_name+'.png',
            dpi=300,bbox_inches = 'tight', papertype = 'A4')
    plt.close()
    return roc_above, roc_below, hrr_above, farr_above, hrr_below, farr_below, hrrd_above, hrrd_below


