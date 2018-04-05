#codigo para calcular la probabilidad pronosticada para cada tercil en cada anio
# a partir de la combinacion de pronosticos con la metodologia de wpdf y wsereg
#modelos calibrados con ereg. Ademas grafica algunos indices de verificacion y guarda los resultados

#!/usr/bin/env python

import argparse #llamado del condigo desde consola
import time #medir tiempo que toma
import numpy as np
import glob 
#import os.path
from pathlib import Path 
from scipy.stats import norm #calculos probabilidades distro normal
from pathos.multiprocessing import ProcessingPool as Pool 
import calendar
import ereg #calibrar superensamble con ensemble regression
import verif_scores #calculo y grafico de los indices de verificacion

cores = 9

def combinar(modelos,instit,varn,latn, lonn, miembros, ic_mes, plazo, sss, fechai,fechaf, p, wtech, ctech, p_mme,
        lat_s,lat_n,lon_oes,lon_est):
    nmodels = len(modelos)
    ny = int(np.abs(lat_n-lat_s)+1)
    nx = int(np.abs (lon_est-lon_oes)+1) #esto no funciona si mi dominio pasa por greenwich
    nyears = int(fechaf[0]-fechai[0]+1)

    #tomo la info de todos los modelos, hay que mejorar la definicion del modelo

    if ctech == 'wpdf':
        prob_terciles = np.array([]).reshape(2,nyears,ny,nx,0) #cat anios lats lons
    else:
        for_dt = np.array([]).reshape(nyears,ny,nx,0)

    if wtech == 'pdf_int':

        weight = np.array([]).reshape(nyears,ny,nx,0)

    elif wtech == 'mean_cor':

        rmean = np.array([]).reshape(ny,nx,0)

    nmembers = np.empty([nmodels],dtype = int)

    for i in np.arange(nmodels):
        
        #abro archivo modelo

        archivo = Path('/datos/osman/nmme_output/cal_forecasts/'+ varn + '_' + modelos[i
            ] +'_'+'{:02}'.format(ic_mes)+'_'+'{:02}'.format(plazo)+'_01_'+
            '{:03}'.format(p)+'_hind.npz')

        if archivo.is_file():

            data = np.load(archivo)

            #extraigo datos de cada modelo. esto depende del tipo de consolidado
            #con wpdf solo necesito la probabilidad y el peso de cada mod
            #con wsereg necesito el prono estandarizado y el peso

            if ctech == 'wpdf':

                #prob acumulada para los limites de cada tercil

                prob_terc = data ['prob_terc']

                prob_terc = np.nanmean(prob_terc,axis = 2)

                nmembers[i] = prob_terc.shape[2]

                prob_terciles = np.concatenate((prob_terciles,prob_terc[:,:,:,:,np.newaxis]), axis = 4)

                prob_terc = []

            else:

                f_dt = data['pronos_dt']

                nmembers[i] = f_dt.shape[1]

                for_dt = np.concatenate((for_dt,np.rollaxis(f_dt,1,4)),axis = 3)

                f_dt = []

            #extraigo info del peso segun la opcion por la que elijo pesar
            if wtech == 'pdf_int': #intensida de la pdf en el punto de la observacion en cada anio
 
                peso = data ['peso']

                weight =np.concatenate((weight,peso[:,0,:,:][:,:,:,np.newaxis]), axis = 3)

                peso = []

            elif wtech == 'mean_cor':

                Rm = data ['Rm']

                rmean = np.concatenate ((rmean, Rm[:,:,np.newaxis]), axis = 2)

    lat = data ['lats']

    lon = data ['lons']
        
    #obtengo datos observados

    archivo = Path('/datos/osman/nmme_output/obs_'+varn+'_'+sss+'.npz')

    data = np.load(archivo)

    obs_terciles = data ['cat_obs']

    if ctech == 'wsereg':
        
        terciles = data['terciles']

        obs_dt = data ['obs_dt']

    
    [ntimes, nlats, nlons] = obs_terciles.shape[1:]

    #defino matriz de peso

    if wtech == 'pdf_int':
        
        #calculo para cada año la probabilidad de cad atercil para cada miembro de ensamble de 
        #cada modelo. Despues saco el promedio

        #peso: nro veces max de la intensidad de la pdf / anios

        maximo = np.ndarray.argmax (weight, axis = 3) #posicion en donde se da el maximo

        weight = []

        peso = np.empty([nmodels, nlats,nlons])

        for i in np.arange(nmodels):
            peso [i,:,:] = np.sum (maximo == i, axis = 0)/ntimes

        peso = np.rollaxis(peso,0,3)

        weight = np.tile (peso,(2,ntimes,1,1,1))

    elif wtech == 'mean_cor':

        rmean [np.where(rmean < 0)] = 0

        rmean [np.where(np.sum(rmean, axis = 2)== 0),: ] = 1

        peso = rmean / np.tile(np.sum(rmean, axis = 2)[:,:,np.newaxis],[1,1,nmodels])
                         
        weight = np.tile   (peso, (2,ntimes,1,1,1))  #2 ntimes nlat nlon nmodels
                         
    elif wtech == 'same': #mismo peso para todos
                          
        weight = np.ones([2,ntimes,nlats,nlons,nmodels])/np.sum(nmodels)
                         
    if ctech == 'wpdf':   
                         
        prob_terc_comb = np.nansum(weight * prob_terciles, axis = 4)
                         
    elif ctech == 'wsereg':                 
        #ereg con el smme pesado

                                 
        pronos_dt = np.rollaxis(for_dt * np.repeat(weight[0,:,:,:,:]/nmembers,nmembers,axis = 3),3,1)      

        [forecast_cr, Rmedio, Rmej, epsb, Kmax, K] = ereg.ensemble_regression(pronos_dt, 
                obs_dt,p_mme, True)
                          
        #calculo la probabilidad de pp para cada tercil, cada anio y cada miembro del modelo
                          
        prob_terc = ereg.probabilidad_terciles (forecast_cr, epsb, terciles)

                        
        #obtengo la combinacion a partir de la suma pesada
                          
        prob_terc_comb = np.nanmean(prob_terc,axis = 2)
        
    #guardo los pronos

    route = '/datos/osman/nmme_output/cal_forecasts/'

    archivo = varn + '_mme_' + '{:02}'.format(ic_mes) + '_' + '{:02}'.format(
            plazo) + '_01_' + '{:03}'.format(p) +'_'+ wtech+'_'+ctech+'_hind.npz'

    np.savez(route+archivo,prob_terc_comb = prob_terc_comb, lat = lat, lon= lon)
                         
    # calculo y grafico los diferentes scores de verificacion:
                          
    bins = np.arange(0,1.1,0.1)
    
    route = '/datos/osman/nmme_figuras/'

    ruta = '/datos/osman/nmme_output/verif_scores/'

    print(prob_terc_comb[:,4,3:5,3:5])

    #RPSS                
    #ploteo rpss         

    rpss = verif_scores.RPSS(prob_terc_comb,obs_terciles)
                         
    titulo = 'Ranked Probabilistic Skill Score'
                         
    salida = route + 'rpss_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
            )+'_'+wtech+'_'+ctech+'_hind.eps'
                         
    verif_scores.plot_scores(lat, lon, rpss, titulo, salida)
   
    archivo = 'rpss_' + varn + '_mme_' + '{:02}'.format(ic_mes) + '_' + '{:02}'.format(
            plazo) + '_01_' + '{:03}'.format(p) +'_'+ wtech+'_'+ctech+'_hind.npz'

    np.savez(ruta+archivo,rpss = rpss, lat = lat, lon= lon)
                         
   #BSS y su descomposicion

    BSS_above = np.empty([6,nlats,nlons])

    for i in np.arange(nlats):
        for j in np.arange(nlons):

            BSS_above[:,i,j] = verif_scores.BS_decomposition(1-prob_terc_comb[1,:,i,j],obs_terciles[2,:,i,j],bins)

    archivo = 'bss_above_' + varn + '_mme_' + '{:02}'.format(ic_mes) + '_' + '{:02}'.format(
            plazo) + '_01_' + '{:03}'.format(p) +'_'+ wtech+'_'+ctech+'_hind.npz'

    np.savez(ruta+archivo,BSS_above = BSS_above, lat = lat, lon= lon)
                     
    titulo = 'Brier Skill Score - Above Normal event'
                         
    salida = route + 'brierss_above_'+varn+'_mme_'+'{:02}'.format(
            ic_mes)+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
            )+'_'+wtech+'_'+ctech+'_hind.eps'
    
    BSS = 1-BSS_above[0]/(0.33*(1-0.33))
    
    verif_scores.plot_scores (lat, lon,BSS , titulo, salida)
                         
    titulo = 'BSS - Resolution - Above Normal event'
                         
    salida = route + 'bss_res_above_'+varn+'_mme_'+'{:02}'.format(
            ic_mes)+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'
                        
    verif_scores.plot_scores (lat, lon, BSS_above[2,:,:], titulo, salida)
                        
    titulo = 'BSS - Reliability - Above Normal event'
                        
    salida = route + 'bss_rel_above_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'
                        
    verif_scores.plot_scores (lat, lon, BSS_above[3,:,:], titulo, salida)
                        
    BSS_below = np.empty([6,nlats,nlons])

    for i in np.arange(nlats):
        for j in np.arange(nlons):

            BSS_below[:,i,j] = verif_scores.BS_decomposition(prob_terc_comb[0,:,i,j],obs_terciles[0,:,i,j],bins)
                         
    archivo = 'bss_below_' + varn + '_mme_' + '{:02}'.format(ic_mes) + '_' + '{:02}'.format(
            plazo) + '_01_' + '{:03}'.format(p) +'_'+ wtech+'_'+ctech+'_hind.npz'

    np.savez(ruta+archivo,BSS_below = BSS_below, lat = lat, lon= lon)

    titulo = 'Brier Skill Score - Below Normal event'
                         
    salida = route + 'brierss_below_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'
    
    BSS = 1-BSS_below[0]/(0.33*(1-0.33))                        
    
    verif_scores.plot_scores (lat, lon, BSS_below[0], titulo, salida)
                         
    titulo = 'BSS - Resolution - Below Normal event'
                         
    salida = route + 'bss_res_below_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'
                         
    verif_scores.plot_scores (lat, lon, BSS_below[2], titulo, salida)
                        
    titulo = 'BSS - Reliability - Below Normal event'
                        
    salida = route + 'bss_rel_below_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'
                         
    verif_scores.plot_scores (lat, lon, BSS_below[3], titulo, salida)
                        
                        
   #AUROC                
                        
    auroc_above = verif_scores.auroc(1-prob_terc_comb[1,:,:,:],obs_terciles[2,:,:,:],lat,bins)
    
    archivo = 'auroc_above_' + varn + '_mme_' + '{:02}'.format(ic_mes) + '_' + '{:02}'.format(
            plazo) + '_01_' + '{:03}'.format(p) +'_'+ wtech+'_'+ctech+'_hind.npz'

    np.savez(ruta+archivo,auroc_above = auroc_above, lat = lat, lon= lon)
                      
    titulo = 'Area under curve ROC - Above Normal event'
                        
    salida = route + 'auroc_above_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'
                        
    verif_scores.plot_scores (lat, lon, auroc_above[:,:], titulo, salida)
                        
    auroc_below = verif_scores.auroc(prob_terc_comb[0,:,:,:],obs_terciles[0,:,:,:],lat,bins)

    archivo = 'auroc_below_' + varn + '_mme_' + '{:02}'.format(ic_mes) + '_' + '{:02}'.format(
            plazo) + '_01_' + '{:03}'.format(p) +'_'+ wtech+'_'+ctech+'_hind.npz'

    np.savez(ruta+archivo,aurco_below = auroc_below, lat = lat, lon= lon)

    titulo = 'Area under curve ROC - Below Normal event'

    salida = route + 'auroc_below_'+varn+'_mme_'+'{:02}'.format(ic_mes
            )+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p
                    )+'_'+wtech+'_'+ctech+'_hind.eps'

    verif_scores.plot_scores (lat, lon, auroc_below[:,:], titulo, salida)

   #Reliability y ROC para:

   #todo el dominio

    salida = varn+'_mme_'+'{:02}'.format(ic_mes)+'_'+'{:02}'.format(plazo
            )+'_01_'+'{:03}'.format(p)+'_'+wtech+'_'+ctech+'_hind_all.eps'

    verif_scores.rel_roc(prob_terc_comb,obs_terciles,lat,bins,route,salida)


    #SA tropical

    #SA extratropical
# ================================================================================
def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Calibrating models using Ensemble Regression.')
    # First arguments. Variable to calibrate. Prec or temp
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable to calibrate (prec or temp)')
    parser.add_argument('IC', type = int, nargs= 1,\
            help = 'Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type = int, nargs = 1,\
            help = 'Forecast leatime (in months, from 1 to 7)')
    parser.add_argument('mod_spread',  type = float, nargs = 1,\
            help = 'percentage of spread to retain in each model (from 0 to 1)')
    parser.add_argument('mme_spread', type = float, nargs = 1,\
            help = 'percentage of spread to retain in the mme (from 0 to 1)') #este argumento deberia
    #existir solo si elijo wsereg como metodo de combinacion. Algo para cambiar en el futuro
    parser.add_argument('--comb-tech',required = True, nargs = 1,\
            choices = ['wpdf','wsereg'],dest = 'ctech', help = 'Combination approach')
    parser.add_argument('--weight-tech', required = True, nargs = 1,\
            choices = ['pdf_int','mean_cor','same'],dest = 'wtech', \
            help = 'Relative weight between models (pdf intensity at obs,mean correlation, same weight)')
    # Specify models to exclude from command line. 
    parser.add_argument('--no-model', required = False, nargs = '+', 
            choices = ['CFSv2', 'CESM1','CanCM3','CanCM4', 'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5'],
            dest ='no_model', help = "Models to be discarded")

    # Extract dates from args

    args=parser.parse_args()

    varn = args.variable[0]

    ic_mes = args.IC[0]

    plazo = args.leadtime[0]

    p = args.mod_spread[0]

    p_mme = args.mme_spread[0]

    wtech = args.wtech[0]

    ctech = args.ctech[0]

    seas = range(ic_mes+plazo,ic_mes+plazo+3)
               
    sss = [i-12 if i>12 else i for i in seas ]

    sss = "".join(calendar.month_abbr[i][0] for i in sss)

    lista = glob.glob("/home/osman/actividades/postdoc/modelos/*")

    modelo = []
    instit = []
    latn = []
    lonn = []
    miembros = []
    plazos = []
    fechai = []
    fechaf = []
    ext = []
                                           
    if args.no_model is not None: #si tengo que descartar modelos

        nombres = args.no_model[:]

        for j in nombres:
            for i in lista:
                lines = [line.rstrip('\n') for line in open(i)]

                modelos = lines[0] ==j

                if modelos: 
                    
                    lista.remove(i)

                    break
    for i in lista:

        lines = [line.rstrip('\n') for line in open(i)]
        
        modelo.append(lines[0])
        
        instit.append(lines[1])
        
        latn.append(lines[2])
        
        lonn.append(lines[3])
        
        miembros.append(int(lines[4]))
        
        plazos.append(int(lines[5]))
        
        fechai.append(int(lines[6]))
        
        fechaf.append(int(lines[7]))
        
        ext.append(lines[8])
        
    combinar(modelo,instit,varn,latn, lonn, miembros, ic_mes, plazo, sss, fechai,fechaf, p, wtech, ctech, p_mme,lat_sur,lat_nor,lon_oes,lon_est) 

#===================================================================================================

start = time.time()
                                                                                                                                                                                                              #abro archivo donde guardo coordenadas
                                                  
coordenadas = 'coords'

lines = [line.rstrip('\n') for line in open(coordenadas)]

lat_sur = float(lines[1])

lat_nor = float(lines [2]) #idem

lon_oes = float(lines[3]) #idem

lon_est = float(lines[4]) #idem

#if __name__=="__main__":
main()

end = time.time()

print(end - start)

# =================================================================================
#start = time.time()

#ic_mes = 11 #en este ejemplo IC Mayo
#plazo = 1 # es este ejemplo prono de Junio-Julio-Agosto
#varn = 'prec'
#sss = 'DJF'
#p = 0.9
#p_mme = 1
#wtech = 'same' #pdf_int or mean_cor or same
#ctech = 'wsereg'  #wpdf or wsereg
#modelo = ['CFSv2','CESM1', 'CanCM3', 'CanCM4', 'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5']
#instit = ['NCEP', 'NCAR', 'CMC','CMC', 'GFDL','GFDL', 'GFDL', 'NASA']
#latn = [ 'Y', 'Y', 'lat','lat', 'Y', 'Y', 'lat','Y' ]
#lonn = [ 'X', 'X', 'lon', 'lon', 'X', 'X', 'lon', 'X']
#miembros = [ 24, 28, 10, 10, 28, 28, 10, 28 ]
#plazos = [ 10, 12, 12, 12, 12, 12, 12, 9 ]
#fechai = [ 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982]
#fechaf = [ 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010]
#ext = ['nc','nc', 'nc4', 'nc4', 'nc', 'nc','nc', 'nc']
#
#
