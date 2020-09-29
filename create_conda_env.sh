#!/usr/bin/env bash

# crear entorno virtual
conda create -n pysol python=3.7
conda activate pysol

# instalar paquetes con pip
python -m pip install numpy
python -m pip install xarray
python -m pip install scipy
python -m pip install astropy
python -m pip install matplotlib
python -m pip install pathos

# instalar paquetes con conda
conda install cartopy
