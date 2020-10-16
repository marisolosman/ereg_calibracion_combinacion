"""
Code to discard Model that is no longer operational
"""
import os
import glob
from datetime import datetime
import pathlib
import configuration

nombre = input("Ingrese nombre del modelo a descartar: ")

if not os.path.isdir("./modelos_viejos"):
    pathlib.Path("./modelos_viejos").mkdir(parents=True, exist_ok=True)

if os.path.isfile(f"./modelos/{nombre}"): 
    os.rename(f"./modelos/{nombre}", f"./modelos_viejos/{nombre}")

with open("updates", "r+") as updatesfile:
    if nombre not in updatesfile.read():
        line_to_add = f"{datetime.now().strftime('%d/%m/%Y')}: Model {nombre} discarded \n"
        updatesfile.write(line_to_add)

#remove combined forecasts
cfg = configuration.Config.Instance()
PATH = f"{cfg.get('download_folder')}/DATA/combined_forecasts/*".replace("//","/")
for f in glob.glob(PATH):
    os.remove(f)
PATH = f"{cfg.get('gen_data_folder')}/nmme_output/comb_forecast/*".replace("//","/")
for f in glob.glob(PATH):
    os.remove(f)
