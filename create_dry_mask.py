"""
Create dry mask where mean seasonal precipitation is less than 1mm/day"""
import datetime
import warnings
import numpy as np
import xarray as xr
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from pandas.tseries.offsets import *
from pathlib import Path
import configuration

cfg = configuration.Config.Instance()

CORES = mp.cpu_count()
PATH = cfg.get('folders').get('download_folder')
ruta = Path(PATH, cfg.get('folders').get('nmme').get('root'))
hind_length = 28

warnings.filterwarnings("ignore", category=RuntimeWarning)

archivo = Path(ruta, 'prec_monthly_nmme_cpc.nc')
coords = cfg.get('coords')
lats = float(coords['lat_s'])
latn = float(coords['lat_n'])
lonw = float(coords['lon_w'])
lone = float(coords['lon_e'])
variable = 'prec'
dataset = xr.open_dataset(archivo)

dataset = dataset.sel(**{'Y': slice(lats, latn), 'X': slice(lonw, lone)})
dataset['T'] = dataset['T'].astype('datetime64[ns]')
#compute 3-month running mean
ds3m = dataset.rolling(T=3, center=True).sum().dropna('T')
#compute climatological mean
ds3m = ds3m.groupby('T.month').mean(skipna=True)
#create dry mask: seasonal precipitation less than 30mm/month
ds3m[variable] = ds3m[variable] <90
PATH = cfg.get('folders').get('gen_data_folder')
ds3m.to_netcdf(Path(PATH, cfg.get('folders').get('data').get('root'), 'dry_mask.nc'))

