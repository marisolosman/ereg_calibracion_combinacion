# About #
My postdoctoral research at [CIMA](http://www.cima.fcen.uba.ar/) (Centro de Investigaciones del Mar y la Atmosfera) by [Sol Osman](https://www.researchgate.net/profile/Marisol_Osman)

Calibration and consolidation of seasonal forecast participating in the [NMME project](http://www.cpc.ncep.noaa.gov.gov/products/NMME/). Forecast can be downloaded through the [IRI DL](iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME)


### About Ensemble Regression ###

Ensemble regression is a technique used to calibrate climate models. Developed by Dave Unger and collaborators. It is used in the Climate Prediction Center to calibrate and consolidate seasonal forecast over North America

### References ###

[Unger, DA and coauthors (2009): Ensemble regresion. Mon Wea Rev, 137, 2365-2379.](https://doi.org/10.1175)

 
## Contents ##

* observation.py
	_manipulates reference database as observation_
* model.py
	_manipulates NMME models_	
* calibration.py 
	_calibrates models using EREG_
* combination.py
	_combines models_
* ereg.py
	_computes enseble regression for multi-model super ensemble_
* plot_forecast.py
	_plots hindcast probabilistic seasonal forecast for temperature and precipitation_
* plot_observed_category.py
	_plots observed seasonal precipitation or temperature category_
* real_time_combination.py
	_computes calibrated real time forecast based on calibrated hindcast parameters_
* plot_rt_forecast.py
	_plots real time precipitation and temperature forecast_
* run_operational_forecast.py
	_calls calibration real_time_combination.py plot_rt_forecast.py_
* run_hindcast_forecast.py
	_call calibration in cross-validated mode calibrates and combines hindcast forecast_

## Requirements ##
There is a ereg.yml file that can be used to generate the environment to implement ereg, and also a bash script (create_conda_env.sh) that create the conda environment and install all requiered python packages
## Usage ##
* Edit the file config.yaml
* Download hindcast and real time forecasts (see examples)
### Hindcast ###
* run calibration with CV mode activated
* run combination with the 6 different approaches
* run plot_forecast
* run plot_observed_category
### Operational Forecast ###
* run calibration without CV mode activated
* run real_time_combination with the 6 different approaches
* run plot_rt_forecast

### NEW MODEL ###
* add model to config.yaml file
* repeat steps for operational forecast

If hindcast forecast are needed you need to remove previous calibrated hindcast forecast and follow the steps for Hincast forecast again

## Way forward ##
* Deal with different ensemble members between IC and between hindcast phase and operational phase (not recommended to preserve performance between hindcast and real-time)
* Improve parser
* Parallelize verification and plotting
* Automatization (check code in Fiona)
