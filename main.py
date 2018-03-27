#codigo sencillo para testear el archivo model.py

#!/usr/bin/env python

import argparse #para hacer el llamado desde consola
import time #tomar el tiempo que lleva el codigo
import numpy as np
import model #objeto y metodos asociados a los modelos
import observation # idem observaciones
import glob #listar archivos
import os.path #manejar path
import calendar #manejar meses del calendario
from pathlib import Path #manejar path

def calibrar(modelos,instit,latn, lonn, miembros, plazos, fechai, fechaf, extension, varn, ic_mes,leadtime,sss, p, lat_sur, lat_nor, lon_oes, lon_est):

    for i in np.arange(len(modelos)):

        #instancio observacion

        archivo = Path('/datos/osman/nmme_output/obs_'+varn+'_'+sss+'.npz')

        output = Path('/datos/osman/nmme_output/'+varn+'_'+ modelos[i] +'_'+
                '{:02}'.format(ic_mes)+'_'+'{:02}'.format(leadtime)+'_01_'+'{:03}'.format(p)+'_hind.npz')

        if archivo.is_file():

            data = np.load(archivo)

            obs_dt = data['obs_dt']

            terciles = data ['terciles']

            data.close()

        else:

            obs = observation.Observ('prec','cpc','prate','Y','X',1982,2011)

            #selecciono trimestre de interes

            print("Selecciono observacion")

            [lats_obs, lons_obs, obs_3m] = obs.select_months(ic_mes,leadtime, lat_sur, lat_nor, lon_oes, lon_est)

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

        #instancio modelo 

        if output.is_file():
            pass
        else:

            modelo = model.Model (modelos[i],instit[i],varn,latn[i],lonn[i],miembros[i],plazos[i],fechai[i],fechaf[i], extension[i])

            print (modelo)

            #tomo los pronos segun la ci y el leadtime

            print("Selecciono pronos")

            [lats,lons,pronos] = modelo.select_months(ic_mes, leadtime, lat_sur, lat_nor, lon_oes, lon_est)
        
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
    parser.add_argument('spread',  type = float, nargs = 1,\
            help = 'percentage of spread to retain (from 0 to 1)')
    # Specify models to exclude from command line. 
    parser.add_argument('--no-model', required = False, nargs = '+', choices = ['CFSv2', 'CESM1','CanCM3','CanCM4', 'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5'],
            dest ='no_model', help="Models to be discarded")

    # Extract dates from args
    args=parser.parse_args()

    varn = args.variable[0]
    
    ic_mes = args.IC[0]
    
    plazo = args.leadtime[0]
    
    p = args.spread[0]

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
        
#llamo a la funcion para calibrar
#def calibrar(modelos,instit,latn, lonn, miembros, plazos, fechai, fechaf, extension, varn, ic_mes,leadtime,sss, p, lat_sur, lat_nor, lon_oes, lon_est):

    calibrar(modelo,instit,latn, lonn, miembros, plazos, fechai,fechaf, ext, varn,ic_mes,plazo, sss, p,lat_sur, lat_nor, lon_oes, lon_est) 
start = time.time()

#abro archivo donde guardo coordenadas
coordenadas = 'coords'

lines = [line.rstrip('\n') for line in open(coordenadas)]

lat_sur = float(lines[1]) #se va a un text file
lat_nor = float(lines [2]) #idem
lon_oes = float(lines[3]) #idem
lon_est = float(lines[4]) #idem

#modelo = ['CFSv2','CESM1', 'CanCM3', 'CanCM4', 'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5']
#instit = ['NCEP', 'NCAR', 'CMC','CMC', 'GFDL','GFDL', 'GFDL', 'NASA']
#latn = [ 'Y', 'Y', 'lat','lat', 'Y', 'Y', 'lat','Y' ]
#lonn = [ 'X', 'X', 'lon', 'lon', 'X', 'X', 'lon', 'X']
#miembros = [ 24, 28, 10, 10, 28, 28, 10, 28 ]
#plazos = [ 10, 12, 12, 12, 12, 12, 12, 9 ]
#fechai = [ 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982]
#fechaf = [ 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010]
#ext = ['nc','nc', 'nc4', 'nc4', 'nc', 'nc','nc', 'nc']
#main(modelo,instit,latn, lonn, miembros, plazos, fechai,fechaf, ext, varn, sss, p) 
if __name__=="__main__":
    main()
end = time.time()

print(end - start)
