"""Function to add new model. Information needed:
    Model Name
    Institution
    Longitude name in netcdf header
    Latitude name in netcdf
    ensemble members in hindcast mode
    beginning of hindcast period
    end of hindcast period
    ensemble members in operational mode
"""
import os
import glob
from datetime import datetime
import configuration

Model = input("Write Model name: ")

if not os.path.isfile(f"./modelos/{Model}"): 
  
    Institution = input("Write Institution name: ")
    Y = input("Write Latitude name: ")
    X = input("Write Longitude name: ")
    while True:
        Members = input("Number of members in hindcast mode: ")
        try:
            Members = int(Members)
        except ValueError:
            print("Numero no valido")
        else:
            break
    while True:
        Leadtimes = input("Number of leadtimes: ")
        try:
            Leadtimes = int(Leadtimes)
        except ValueError:
            print("Numero no valido")
        else:
            break
    while True:
        Hindcast_begin = input("Year begining hindcast period: ")
        try:
            Hindcast_begin = int(Hindcast_begin)
        except ValueError:
            print("Numero no valido")
        else:
            break
    while True:
        Hindcast_end = input("Year end hindcast period: ")
        try:
            Hindcast_end = int(Hindcast_end)
        except ValueError:
            print("Numero no valido")
        else:
            break
    
    File_end = input("Enter end of forecast file (nc or nc4)")
    
    while True:
        Members_rt = input("Number of members in hindcast mode: ")
        try:
            Members_rt = int(Members_rt)
        except ValueError:
            print("Numero no valido")
        else:
            break
    
    with open(f"./modelos/{Model}", "w") as modelfile:
        modelfile.write(Model + "\n")
        modelfile.write(Institution + "\n")
        modelfile.write(str(Members) + "\n")
        modelfile.write(str(Leadtimes) + "\n")
        modelfile.write(str(Hindcast_begin) + "\n")
        modelfile.write(str(Hindcast_end) + "\n")
        modelfile.write(File_end + "\n")
        modelfile.write(str(Members_rt))
    
with open("updates", "r+") as updatesfile:
    if Model not in updatesfile.read():
        line_to_add = f"{datetime.now().strftime('%d/%m/%Y')}: Model {Model} added \n"
        updatesfile.write(line_to_add)

#remove combined forecasts
cfg = configuration.Config.Instance()
PATH = f"{cfg.get('download_folder')}/DATA/combined_forecasts/*".replace("//","/")
for f in glob.glob(PATH):
    os.remove(f)
PATH = f"{cfg.get('gen_data_folder')}/nmme_output/comb_forecast/*".replace("//","/")
for f in glob.glob(PATH):
    os.remove(f)
