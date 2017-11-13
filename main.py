#codigo sencillo para testear el archivo model.py

#!/usr/bin/env python

import argparse 
import time
import model
import observation
#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
import glob
import os.path
start = time.time()
lat_sur = -60
lat_nor = 15
lon_oes = 265
lon_est = 330
#instancio un modelo de ejemplo
def main():
    cfs = model.Model ('CFSv2','NCEP','prec','Y','X',24,10,1982,2011)

    print(cfs)

    #tomo los pronos segun la ci y el leadtime

    print("Selecciono pronos")

    [lats,lons,pronos] = cfs.select_months(3,2, lat_sur, lat_nor, lon_oes, lon_est)
    print("Remuevo tendencia")

    pronos_dt = cfs.remove_trend(pronos, True)

    #tengo que escribir aca codigo para estandarizar observaciones en CV
    
    #instancio observacion

    obs = observation.Observ('prec','cpc','prate','Y','X',1982,2011)

    #selecciono trimestre de interes

    print("Selecciono observacion")


    [lats_obs, lons_obs, obs_3m] = obs.select_months(3,2, lat_sur, lat_nor, lon_oes, lon_est)

    #estandarizo observacion
    print("Remuevo tendencia obs")


    obs_dt = obs.remove_trend(obs_3m, True)

    print(obs_dt.shape)

    #aplico ereg con cada modelo el imput es el modelo y las observaciones estan

    #darizadas mientras que el output es el valor y' Kmax admitido espb Rmean

    print("aplico ensemble regression")

    
    [forecast_cr, Rmedio, Rmej, epsb, Kmax] = cfs.ereg(pronos_dt, obs_dt, True)

    #aca deberia ir el paso del analisis de la dispersion del ensamble para ver
    #si tengo que ajustar el ensamble con el k estimado. opciones:

    #si Rmej <=1 => K = 1  NO TOCO EL SPREAD DEL ENSAMBLE
    #si Rmej >1 => uso K <=> K<1 si no, K=0

    #YA HICE EL AJUSTE A LA HORA DE APLICAR EREG YA QUE NO HACE FALTA APLICARLO
    #PARA SABER SI TENGO QUE AJUSTAR EL ENSAMBLE O NO

    #ahora deberia pesar cada modelo de acuerdo a la historia previa. genero
    #un metodo para evaluar la pdf en el valor de la observacion

    print("estimo peso del modelo")

    pdf_intensity = cfs.pdf_eval(forecast_cr,epsb,obs_dt)

    #ahora deberia pesar todos los modelos y guardar los modelos calibrados
    #

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
main() 
end = time.time()
print(end - start)
