# About #
My postdoctoral research at [CIMA](http://www.cima.fcen.uba.ar/) (Centro de Investigaciones del Mar y la Atmosfera) by [Sol Osman](https://www.researchgate.net/profile/Marisol_Osman)

Calibration and consolidation of seasonal forecast participating in the [NMME project](http://www.cpc.ncep.noaa.gov.gov/products/NMME/). Forecast can be downloaded through the [IRI DL](iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME)


### About Ensemble Regression ###

Ensemble regression is a technique used to calibrate climate models. Developed by Dave Unger and collaborators. It is used in the Climate Prediction Center to calibrate and consolidate seasonal forecast over North America

### References ###

[Unger, DA and coauthors (2009): Ensemble regresion. Mon Wea Rev, 137, 2365-2379.](https://doi.org/10.1175)

 
## Contents ##

* observation.py
	_methods to manipulate reference database as observation_
* model.py
	_methods to manipulate NMMe models_	
* calibration.py 
	_calibrates models using EREG_
* combination.py
	_combines models_
* ereg.py
	_computes enseble regression for multi-model super ensemble_
* run_hindcast_forecast.sh
	_calibrates and combines hindcast forecast_
* plot_forecast.py
	_plots hindcast probabilistic seasonal forecast for temperature and precipitation_
* plot_observed_category.py
	_plots observed seasonal precipitation or temperature category_
* real_time_combination.py
	_computes calibrated real time forecast based on calibrated hindcast parameters_
* plot_rt_forecast.py
	_plots real time precipitation and temperature forecast_
* run_operational_forecast.sh
	_calls real_time_combination.py plot_rt_forecast.py_

## Requirements ##

## Usage ##

#Hindcast#

#Operational Forecast#
* Obtain observed and forecasted parameter using calibration without CV option (15 min if parameters have not been obtained yet, else 12sec)
* Run operational combination forecast (40sec each)
* Plot forecast (36sec each)

## Way forward ##
* clean codes: Remove prints and improve dealing with nans
* Deal with different ensemble members between IC and between hindcast phase and operational phase
* Improve parser
* Include changes in model spread
* Parallelize verification and plotting
* Automatization
