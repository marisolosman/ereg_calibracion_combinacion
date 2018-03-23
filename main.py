#codigo sencillo para testear el archivo model.py

#!/usr/bin/env python

import argparse 
import time
import numpy as np
import model
import observation
import glob
import os.path
from pathlib import Path 

start = time.time()

lat_sur = -10
lat_nor = 0
lon_oes = 295
lon_est = 300
ic_mes = 6 #en este ejemplo IC Nov
plazo = 1 # es este ejemplo prono de Dec-Jan-Feb
p = 0.95 #proporcion del spread maximo que quiero retener
varn = 'prec'
sss = 'JJA'
def main(modelos,instit,latn, lonn, miembros, plazos, fechai, fechaf, extension, varn, sss, p):

    for i in np.arange(len(modelos)):

        #instancio observacion

        archivo = Path('/datos/osman/nmme_output/obs_'+varn+'_'+sss+'.npz')

        output = Path('/datos/osman/nmme_output/'+varn+'_'+ modelos[i] +'_'+
                '{:02}'.format(ic_mes)+'_'+'{:02}'.format(plazo)+'_01_'+'{:03}'.format(p)+'_hind.npz')

        if archivo.is_file():

            data = np.load(archivo)

            obs_dt = data['obs_dt']

            terciles = data ['terciles']

            data.close()

        else:

            obs = observation.Observ('prec','cpc','prate','Y','X',1982,2011)

            #selecciono trimestre de interes

            print("Selecciono observacion")

            [lats_obs, lons_obs, obs_3m] = obs.select_months(ic_mes,plazo, lat_sur, lat_nor, lon_oes, lon_est)

            #remuevo tendencia y estandarizo observacion

            print("Remuevo tendencia obs")

            obs_dt = obs.remove_trend(obs_3m, True) #la variable logica habilita o no CV

            print(obs_dt.shape)

            #obtengo los limites de los terciles

            print("Calculo terciles")

            terciles = obs.computo_terciles(obs_dt,True) # la variable logica habilita o no CV


            print("Calculo que tercil se observo en cada anio")

            categoria_obs = obs.computo_categoria (obs_dt, terciles)

            np.savez(archivo, obs_dt = obs_dt,
                    lats_obs = lats_obs, lons_obs = lons_obs, terciles = terciles, cat_obs =
                    categoria_obs)

        #instancio un modelo de ejemplo

        if output.is_file():
            pass
        else:

            modelo = model.Model (modelos[i],instit[i],varn,latn[i],lonn[i],miembros[i],plazos[i],fechai[i],fechaf[i], extension[i])

            print (modelo)

            #tomo los pronos segun la ci y el leadtime

            print("Selecciono pronos")

            [lats,lons,pronos] = modelo.select_months(ic_mes, plazo, lat_sur, lat_nor, lon_oes, lon_est)
        
            print("Remuevo tendencia")

            pronos_dt = modelo.remove_trend(pronos, True)

            #aplico ereg:  el imput es el modelo y las observaciones estandarizadas y la proporcion
            #de kmax a aplicar en caso de correccion de ensamble
            #mientras que el output es el valor y' Kmax admitido espb Rmean

            print("aplico ensemble regression")

            [forecast_cr, Rmedio, Rmej, epsb, Kmax, K] = modelo.ereg(pronos_dt, obs_dt,p, True)

            #incluye parametro p entre 0 y 1 que representa la proporcion del factor de inflacion kmax
            #que le aplico al ensamble, siguiendo a unger et al 2008
            
            #calculo la probabilidad de pp para cada tercil, cada anio y cada miembro del modelo

            prob_terc = modelo.probabilidad_terciles (forecast_cr, epsb, terciles)
            
            #ahora deberia pesar cada modelo de acuerdo a la historia previa. 
            #genero un metodo para evaluar la pdf en el valor de la observacion
            #OJO SOLO ESTOY PROGRAMANDO UN TIPO DE PESO PORQUE EL OTRO ESTA BASADO EN Rmedio QUE YA 
            #LO OBTUVE

            #creo que chace the rapper puede gustarme, pero te odio lana del rey
            print("estimo peso del modelo")

            pdf_intensity = modelo.pdf_eval(forecast_cr,epsb,obs_dt)

            #guardo el pronostico calibrado y la intensidad de la pdf

            np.savez(output,lats = lats, lons = lons, pronos_dt = pronos_dt, pronos_cr = forecast_cr, eps = epsb,Rm = Rmedio, Rb = Rmej, K2 = Kmax, 
                    K = K, peso = pdf_intensity, prob_terc = prob_terc)

#def plot_data_matrix(data_matrix):
#    pass


# Clean working environment (should be called before main)
#def clean():
#    import os
#    os.system("rm -f /tmp/*.gz /tmp/*.nc")

#def main():
  
	# Define parser data
#    parser = argparse.ArgumentParser(description='Plotting satellite data.')
    # First arguments. Dates. TODO:SPECIFY INITIAL AND FINAL ORDER
#    parser.add_argument('date', metavar='YYYY.MM.DD YYYY.MM.DD', type=str, nargs=2,\
#		      help='Initial date followed by end date')
#    # Specify sattelites to exclude from command line. TODO: change to flag!
#    parser.add_argument('--no-ascat', dest='ascat_bool', action="store_true", \
#		      default= False, help="Don't display ASCAT information")

    # Extract dates from args
#    args=parser.parse_args()
#    initialDate = args.date[0]
#    finalDate = args.date[1]
    
    # Flow control depending on specified options
#    if not args.ascat_bool:
        # Instantiate ASCAT and get datetime object
#        ascat = satellite.ASCAT(initialDate, finalDate)
#        ascat.get_datetime_object()
        # Download files from ASCAT servers
#        ascat.download_files()
        # Get figure handler and colormap
#        m, cmap = satellite.generate_figure()   
        # Process for every *.nc file in folder
#        for src_name in glob.glob(os.path.join("/tmp", '*.nc')):
#            base = os.path.basename(src_name)
#            lat, lon, data = ascat.extract_data(src_name)
#            satellite.plot_data(m, lat, lon, data, cmap)
        # Finalize plot design and show it
#        plt.title('ASCAT - Sfc Wind Speed')
#        plt.show()

#    else:
#        print "You've discarded all the satellites I know!"

#if __name__ == "__main__":
#    clean()
modelo = ['CFSv2','CESM1', 'CanCM3', 'CanCM4', 'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5']
instit = ['NCEP', 'NCAR', 'CMC','CMC', 'GFDL','GFDL', 'GFDL', 'NASA']
latn = [ 'Y', 'Y', 'lat','lat', 'Y', 'Y', 'lat','Y' ]
lonn = [ 'X', 'X', 'lon', 'lon', 'X', 'X', 'lon', 'X']
miembros = [ 24, 28, 10, 10, 28, 28, 10, 28 ]
plazos = [ 10, 12, 12, 12, 12, 12, 12, 9 ]
fechai = [ 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982]
fechaf = [ 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010]
ext = ['nc','nc', 'nc4', 'nc4', 'nc', 'nc','nc', 'nc']
main(modelo,instit,latn, lonn, miembros, plazos, fechai,fechaf, ext, varn, sss, p) 

end = time.time()
print(end - start)
