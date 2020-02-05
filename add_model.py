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

Model = input("Write Model name: ")
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

file1 = open("./modelos/" + Model, "w")
file1.write(Model + "\n")
file1.write(Institution + "\n")
file1.write(str(Members) + "\n")
file1.write(str(Leadtimes) + "\n")
file1.write(str(Hindcast_begin) + "\n")
file1.write(str(Hindcast_end) + "\n")
file1.write(File_end + "\n")
file1.write(str(Members_rt))
file1.close()

file1 = open("updates", "a")
file1.write(datetime.now().strftime("%d/%m/%Y") + ": Model " + Model + " addedd \n")

#remove combined forecasts
file1 = open("configuracion", 'r')
PATH = file1.readline()
file1.close()
files = glob.glob(PATH + 'DATA/combined_forecasts/*')
for f in files:
    os.remove(f)



