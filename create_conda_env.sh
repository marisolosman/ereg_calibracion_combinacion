#!/usr/bin/env bash

# crear entorno virtual
conda create -n pysol python=3.7
conda activate pysol

# instalar paquetes en el OS
conda install -c conda-forge cdo
conda install -c conda-forge nco

# instalar paquetes con pip
python -m pip install numpy
python -m pip install xarray
python -m pip install scipy
python -m pip install astropy
python -m pip install matplotlib
python -m pip install pathos
python -m pip install netcdf4
python -m pip install cdo
python -m pip install nco
python -m pip install python-crontab
python -m pip install PyYAML

# instalar paquetes con conda
conda install -c conda-forge cartopy
