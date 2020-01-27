"""
Code to discard Model that is no longer operational
"""
import os
from datetime import datetime

nombre = input("Ingrese nombre del modelo a descartar: ")

os.rename('./modelos/' + nombre, './modelos_viejos/' + nombre)


file1 = open("updates", "a")
file1.write(datetime.now().strftime("%d/%m/%Y") + ": Model " + nombre + "discarded \n")
