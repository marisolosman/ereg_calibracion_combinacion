"""
Code to discard Model that is no longer operational
"""
import os
import glob
from datetime import datetime

nombre = input("Ingrese nombre del modelo a descartar: ")
file1 = open("configuracion", 'r')
PATH = file1.readline().rstrip('\n')

os.rename(PATH + 'modelos/' + nombre, PATH + 'modelos_viejos/' + nombre)


file1 = open("updates", "a")
file1.write(datetime.now().strftime("%d/%m/%Y") + ": Model " + nombre + " discarded \n")
#remove combined forecasts
file1.close()
files = glob.glob(PATH + 'DATA/combined_forecasts/*')
for f in files:
    os.remove(f)


