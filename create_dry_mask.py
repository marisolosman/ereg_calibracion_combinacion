"""
Create dry mask where mean seasonal precipitation is less than 1mm/day"""
import datetime
import warnings
import numpy as np
import xarray as xr
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from pandas.tseries.offsets import *
CORES = mp.cpu_count()
file1 = open('configuracion', 'r')
PATH = file1.readline().rstrip('\n')
file1.close()
ruta = PATH + 'NMME/'
archivo = ruta + 'prec_monthly_nmme_cpc.nc'
hind_length = 28
warnings.filterwarnings("ignore", category=RuntimeWarning)
coordenadas = 'coords'
domain = [line.rstrip('\n') for line in open(coordenadas)]  #Get domain limits
lats = float(domain[1])
latn = float(domain[2])
lonw = float(domain[3])
lone = float(domain[4])
variable = 'prec'
#hay problemas para decodificar las fechas, genero un xarray con mis fechas decodificadas
dataset = xr.open_dataset(archivo)

dataset = dataset.sel(**{'Y': slice(lats, latn), 'X': slice(lonw, lone)})
dataset['T'] = dataset['T'].astype('datetime64[ns]')


#numero = [int(s) for s in dataset.T.units.split() if s.isdigit()]

#pivot = datetime.datetime(1960, 1, 1) #dificilmente pueda obtener este atributo del nc sin
#poder decodificarlo con xarray
#time = [pivot + DateOffset(months=int(x), days=15) for x in dataset['T']]
#genero xarray con estos datos para obtener media estacional
#ds = xr.Dataset({variable: (('time', 'Y', 'X'), var_out)},
#                coords={'time': time, 'Y': lat, 'X': lon})
#compute 3-month running mean
ds3m = dataset.rolling(T=3, center=True).sum().dropna('T')
#compute climatological mean
ds3m = ds3m.groupby('T.month').mean(skipna=True)
#create dry mask: seasonal precipitation less than 30mm/month
ds3m[variable] = ds3m[variable] <90
print(ds3m)
ds3m.to_netcdf(PATH + 'DATA/dry_mask.nc')
