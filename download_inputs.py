#!/usr/bin/env python

import argparse #parse command line options
import calendar
import helpers
import configuration
import pandas as pd
import urllib.request
import urllib.parse
import xarray as xr
import os.path
import datetime
import time
import yaml
import logging
import sys


cfg = configuration.Config.Instance()

RUTA_IRI = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/"

with helpers.localized("en_US.utf8"):
  MONTHS_ABBR = list(calendar.month_abbr)


def generate_download_url(variable, forecast_start_year, forecast_start_month_abbr, member_or_realization, iridl_model_name):
  # Forecast Start Time (forecast_reference_time)
  #   grid: /S (months since 1960-01-01) ordered (0000 1 Feb 1981) to (0000 1 Jan 2017) by 1.0 N= 432 pts :grid
  forecast_start_time = f"(0000 1 {forecast_start_month_abbr} {forecast_start_year} )"
  # Ensemble Member (realization)
  #   grid: /M (unitless) ordered (1.0) to (x.0) by 1.0 N= x pts :grid
  ensemble_member = f"({member_or_realization}.0 )"
  return f"{RUTA_IRI}.{iridl_model_name}/.HINDCAST/.MONTHLY/.{variable}/S/{forecast_start_time}VALUES/M/{ensemble_member}VALUES/data.nc"


def generate_filename(model_name, model_institution, variable, year, month, member):
  #
  m_str = str(month).zfill(2)
  forecast_month = month - 1 if month > 1 else 12 
  forecast_year = year if forecast_month == 12 else year + 1
  fm_str = str(forecast_month).zfill(2)
  return f"{variable}_Amon_{model_institution}-{model_name}_{year}{m_str}_r{member}_{year}{m_str}-{forecast_year}{fm_str}.nc"


def check_file(filename, recheck=False):
  #
  if not os.path.isfile(filename):
    return False
  if recheck:
    # Check file size
    if os.stat(filename).st_size == 0:
      return False
    # Check if file can be opened
    if filename.endswith('.nc'):
      try:
        d = xr.open_dataset(filename, decode_times=False)
        d.close()
      except Exception as e:
        return False
  return True


def links_to_download_hindcast(df_modelos, recheck):
  # 
  for model_data in df_modelos.itertuples():
    for variable in ["tref", "prec"]:
      for member in range(1, model_data.members+1, 1): 
        for year in range(model_data.hindcast_begin, model_data.hindcast_end+1, 1):
          for month, month_abbr in zip(range(1,13), MONTHS_ABBR[1:]):
            FOLDER = f"{cfg.get('download_folder')}/NMME/hindcast/".replace("//","/")
            DOWNLOAD_URL = generate_download_url(variable, year, month_abbr, member, model_data.iridl_model_name)
            FILENAME = generate_filename(model_data.model, model_data.institution, variable, year, month, member)
            yield {'FILENAME': FOLDER+FILENAME, 'DOWNLOAD_URL': DOWNLOAD_URL, 
                   'DOWNLOADED': check_file(FOLDER+FILENAME, recheck), 'TYPE': 'hindcast'}


def links_to_download_operational(df_modelos, year, recheck):
  # 
  now = datetime.datetime.now()
  for model_data in df_modelos.itertuples():
    for variable in ["tref", "prec"]:
      for member in range(1, model_data.members+1, 1): 
        for month, month_abbr in zip(range(1, now.month+1 if year == now.year else 13), MONTHS_ABBR[1:]):
          FOLDER = f"{cfg.get('download_folder')}/NMME/real_time/".replace("//","/")
          DOWNLOAD_URL = generate_download_url(variable, year, month_abbr, member, model_data.iridl_model_name)
          FILENAME = generate_filename(model_data.model, model_data.institution, variable, year, month, member)
          yield {'FILENAME': FOLDER+FILENAME, 'DOWNLOAD_URL': DOWNLOAD_URL, 
                 'DOWNLOADED': check_file(FOLDER+FILENAME, recheck), 'TYPE': 'operational'}


def links_to_download_real_time(df_modelos, year, month, recheck):
  # 
  for model_data in df_modelos.itertuples():
    for variable in ["tref", "prec"]:
      for member in range(1, model_data.members+1, 1): 
        month_abbr = MONTHS_ABBR[month]
        FOLDER = f"{cfg.get('download_folder')}/NMME/real_time/".replace("//","/")
        DOWNLOAD_URL = generate_download_url(variable, year, month_abbr, member, model_data.iridl_model_name)
        FILENAME = generate_filename(model_data.model, model_data.institution, variable, year, month, member)
        yield {'FILENAME': FOLDER+FILENAME, 'DOWNLOAD_URL': DOWNLOAD_URL, 
               'DOWNLOADED': check_file(FOLDER+FILENAME, recheck), 'TYPE': 'real_time'}


def links_to_download_observation(recheck):
  #
  FOLDER = f"{cfg.get('download_folder')}/NMME/hindcast/".replace("//","/")
  #
  FILENAME = "prec_monthly_nmme_cpc.nc"
  DOWNLOAD_URL = f"{RUTA_IRI}.CPC-CMAP-URD/.prate/data.nc"
  yield {'FILENAME': FOLDER+FILENAME, 'DOWNLOAD_URL': DOWNLOAD_URL, 
         'DOWNLOADED': check_file(FOLDER+FILENAME, recheck), 'TYPE': 'observation'}
  #
  FILENAME = "tref_monthly_nmme_ghcn_cams.nc"
  DOWNLOAD_URL = f"{RUTA_IRI}.GHCN_CAMS/.updated/data.nc"
  yield {'FILENAME': FOLDER+FILENAME, 'DOWNLOAD_URL': DOWNLOAD_URL, 
         'DOWNLOADED': check_file(FOLDER+FILENAME, recheck), 'TYPE': 'observation'}
  #
  FILENAME = "lsmask.nc"
  DOWNLOAD_URL = f"{RUTA_IRI}.LSMASK/.land/data.nc"
  yield {'FILENAME': FOLDER+FILENAME, 'DOWNLOAD_URL': DOWNLOAD_URL, 
         'DOWNLOADED': check_file(FOLDER+FILENAME, recheck), 'TYPE': 'observation'}


def modify_observation_files():
  #
  FOLDER = f"{cfg.get('download_folder')}/NMME/hindcast/".replace("//","/")
  #
  FILENAME = f"{FOLDER}/prec_monthly_nmme_cpc.nc".replace("//","/")
  if os.path.isfile(FILENAME):
    pass
  #
  FILENAME = f"{FOLDER}/tref_monthly_nmme_ghcn_cams.nc".replace("//","/")
  if os.path.isfile(FILENAME):
    pass


def download_file(download_url, filename):
  #
  download_url = urllib.parse.quote(download_url, safe=':/')
  # Download file
  #f, h = urllib.request.urlretrieve(download_url, filename)
  # Check file size
  #assert os.stat(file_name).st_size != 0
  # Check if file can be opened
  #if filename.endswith('.nc'):
  #  d = xr.open_dataset(file_name, decode_times=False)
  #  d.close()


# ==================================================================================================
if __name__ == "__main__":
  now = datetime.datetime.now()
  
  # PROCESAR ARGUMENTOS
  parser = argparse.ArgumentParser(description='Download input data')
  parser.add_argument('--download', nargs='+', default=['all'],
    choices=['hindcast','operational','real_time','observation','all'], 
    help='Indicates which input data should be downloaded')
  parser.add_argument('--year', type=int, default=now.year,
    help='Indicates input data of which years should be downloaded for operational and real-time execution')
  parser.add_argument('--month', type=int, default=now.month,
    help='Indicates input data of which months should be downloaded for real-time execution')
  parser.add_argument('--recheck', type=bool, default=True,
    help='Indicates if previously downloaded files must be checked or not')
  parser.add_argument('--recheck-hindcast', type=bool, default=False, dest='recheck_hindcast',
    help='Indicates if previously downloaded hindcasts files must be checked or not')
  args = parser.parse_args()  # Extract dates from args
  # args = argparse.Namespace(download=['all'], year=2020, month=6, re-check=False)
  
  
  # INFORMAR SOBRE VERIFICACIÓN DE ARCHIVOS
  cfg.logger.info(f'Previously downloaded files will{" " if args.recheck else " not "}be verified!')
    
  
  # IDENTIFICAR MODELOS A SER UTILIZADOS
  conf_modelos = cfg.get('models')
  df_modelos = pd.DataFrame(conf_modelos[1:], columns=conf_modelos[0])
  
  # GENERAR LINKS DE DESCARGA
  df_links = pd.DataFrame(columns=['FILENAME','DOWNLOAD_URL','DOWNLOADED','TYPE'])
  if any(item in ['hindcast', 'all'] for item in args.download):
    start = time.time()
    links = links_to_download_hindcast(df_modelos, args.recheck_hindcast)
    df_links = df_links.append(pd.DataFrame.from_dict(links), ignore_index=True)
    end = time.time()
    cfg.logger.info(f'Time to gen{" and recheck " if args.recheck_hindcast else " "}hindcast links: {end - start}')
  if any(item in ['operational', 'all'] for item in args.download):
    start = time.time()
    links = links_to_download_operational(df_modelos, args.year, args.recheck)
    df_links = df_links.append(pd.DataFrame.from_dict(links), ignore_index=True)
    end = time.time()
    cfg.logger.info(f'Time to gen{" and recheck " if args.recheck else " "}operational links: {end - start} -> anho: {args.year}')
  if any(item in ['real_time', 'all'] for item in args.download):
    start = time.time()
    links = links_to_download_real_time(df_modelos, args.year, args.month, args.recheck)
    df_links = df_links.append(pd.DataFrame.from_dict(links), ignore_index=True)
    end = time.time()
    cfg.logger.info(f'Time to gen{" and recheck " if args.recheck else " "}real_time links: {end - start} -> anho: {args.year}, mes: {args.month}')
  if any(item in ['observation', 'all'] for item in args.download):
    start = time.time()
    links = links_to_download_observation(args.recheck)
    df_links = df_links.append(pd.DataFrame.from_dict(links), ignore_index=True)
    end = time.time()
    cfg.logger.info(f'Time to gen{" and recheck " if args.recheck else " "}observation links: {end - start}')
  
  total_files = df_links['DOWNLOADED'].count()
  downloaded_files = df_links['DOWNLOADED'].value_counts()
  n_downloaded_files = downloaded_files[True]
  n_files_to_download = downloaded_files[False]
  cfg.logger.info(f"Total files: {total_files}, "+
                  f"Downloaded files: {n_downloaded_files}, "+
                  f"Not yet downloaded files: {n_files_to_download}")
  
  # DESCARGAR ARCHIVOS
  count_downloaded_files = 0
  for row in df_links.query('DOWNLOADED == False').itertuples():
    if not check_file(row.FILENAME, True):
      try:
        download_file(row.DOWNLOAD_URL, row.FILENAME)
        count_downloaded_files += 1
      except Exception as e:
        cfg.logger.error(e)
        cfg.logger.warning(f'Failed to download file "{row.FILENAME}" from url "{row.DOWNLOAD_URL}"')
        continue
      else:
        df_links.at[row.Index, 'DOWNLOADED'] = True
        configuration.progress(count_downloaded_files, n_files_to_download, status='Downloading files')
  configuration.close_progress()
  cfg.logger.info(f'{count_downloaded_files} files were downloaded successfully and '+
                  f'{n_files_to_download - count_downloaded_files} downloads failed!')

  if any(item in ['observation', 'all'] for item in args.download):
    modify_observation_files()
