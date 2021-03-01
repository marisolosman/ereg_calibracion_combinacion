#!/usr/bin/env python

# OBS:
# El resultado de la calibración sin cross-validation se usa para la combinación en tiempo real "real_time_combination.py"
# El resultado de la calibración con cross-validation se usa para la combinación general "combination.py"

import argparse  # parse command line options
import time  # test time consummed
import configuration
import itertools

from calibration import main as calibration
from combination import main as combination
from plot_forecast import main as plot_forecast
from plot_observed_category import main as plot_observed_category

cfg = configuration.Config.Instance()

def main(args):
  
    if args.calibrate:
        cfg.logger.info("Starting calibration")
        for v in args.variables:  # loop sobre las variables a calibrar
            for m in range(1, 12+1):  # loop over IC --> Month of initial conditions (from 1 for Jan to 12 for Dec)
                for l in range(1, 7+1):  # loop over leadtime --> Forecast leadtime (in months, from 1 to 7)
                    calibration(argparse.Namespace(variable=[v], IC=[m], leadtime=[l], CV=True, OW=args.overwrite,
                                                   no_models=args.no_models, models=args.models))

    if args.combine:
        cfg.logger.info("Starting combination")
        for v in args.variables:  # loop sobre las variables a calibrar
            for m in range(1, 12+1):  # loop over IC --> Month of initial conditions (from 1 for Jan to 12 for Dec)
                for l in range(1, 7+1):  # loop over leadtime --> Forecast leadtime (in months, from 1 to 7)
                    for c, w in itertools.product(args.combination, args.weighting): 
                        if c == 'count' and w != 'same':
                            continue
                        combination(argparse.Namespace(variable=[v], IC=[m], leadtime=[l], no_models=[], ctech=c, wtech=[w]))
  
    if args.plot:
        cfg.logger.info("Starting plotting")
        for v in args.variables:  # loop sobre las variables a calibrar
            for m in range(1, 12+1):  # loop over IC --> Month of initial conditions (from 1 for Jan to 12 for Dec)
                for l in range(1, 7+1):  # loop over leadtime --> Forecast leadtime (in months, from 1 to 7)
                    plot_forecast(argparse.Namespace(variable=[v], IC=[m], leadtime=[l]))
                plot_observed_category(argparse.Namespace(variable=[v], IC=[m]))
                

# ==================================================================================================
if __name__ == "__main__":
  
    # Defines parser data
    parser = argparse.ArgumentParser(description='Run hindcast forecast')
    groupm = parser.add_mutually_exclusive_group()
    groupm.add_argument('--models', nargs='+', dest='models',
        default=[], choices=[item[0] for item in cfg.get('models')[1:]], 
        help='Indicates which models should be considered (only used for calibration purposes).')
    groupm.add_argument('--no-models', nargs='+', dest='no_models', 
        default=[], choices=[item[0] for item in cfg.get('models')[1:]], 
        help='Indicates which models should be excluded (only used for calibration purposes).')
    parser.add_argument('--variables', nargs='+', 
        default=["tref", "prec"], choices=["tref", "prec"],
        help='Variables that will be considered in the forecast generation process.')
    parser.add_argument('--weighting', nargs='+', 
        default=["same", "pdf_int", "mean_cor"], choices=["same", "pdf_int", "mean_cor"],
        help='Weighting methods used when combining models.')
    parser.add_argument('--combination', nargs='+', 
        default=["wpdf", "wsereg", "count"], choices=["wpdf", "wsereg", "count"],
        help='Combination methods (count will be ignored when calibration is set as operational).')
    parser.add_argument('--overwrite', action='store_true', 
        help='Indicates if previous generated files should be overwrite or not.')
    parser.add_argument('--ignore-calibration', action='store_false', dest='calibrate', 
        help='Indicates if the calibration step should be ignored or not.')
    parser.add_argument('--ignore-combination', action='store_false', dest='combine', 
        help='Indicates if the combination step should be ignored or not.')
    parser.add_argument('--ignore-plotting', action='store_false', dest='plot', 
        help='Indicates if the plotting step should be ignored or not.')

    # Extract data from args
    args = parser.parse_args()
    
    # Run hindcast forecast
    start = time.time()
    try:
        main(args)
    except Exception as e:
        error_detected = True
        cfg.logger.error(f"Failed to run \"run_hindcast_forecast.py\". Error: {e}.")
        raise  # see: http://www.markbetz.net/2014/04/30/re-raising-exceptions-in-python/
    else:
        error_detected = False
    finally:
        end = time.time()
        err_pfx = "with" if error_detected else "without"
        message = f"Total time to run \"run_hindcast_forecast.py\" ({err_pfx} errors): {end - start}" 
        print(message) if not cfg.get('use_logger') else cfg.logger.info(message)