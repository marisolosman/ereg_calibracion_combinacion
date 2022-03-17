
import configuration

import argparse
import time
import calendar
import datetime
import os

from pathlib import Path
from typing import TextIO

cfg = configuration.Config.Instance()


def write_file_desc(fp_file: TextIO, fcst_file_type: str, fcst_file_path: Path):
    fp_file.write('  - {\n')
    fp_file.write(f'    type: "{fcst_file_type}",\n')
    fp_file.write(f'    path: ".",\n')
    fp_file.write(f'    name: "{fcst_file_path.name}",\n')
    if "_hind.npz" in fcst_file_path.name:
        fp_file.write(f'    first_year_in_file: 1982,\n')
    fp_file.write('  }\n')


def realtime_output_filename_first_section(v, ic, lt):
    # Defino ref dataset y target season
    init_date = datetime.datetime.strptime(ic, '%Y-%m-%d')
    init_year = init_date.year
    init_month = init_date.month
    init_month_abbr = calendar.month_abbr[init_month]
    seas = range(init_month + lt, init_month + lt + 3)
    season = "".join(calendar.month_abbr[i][0] for i in [i - 12 if i > 12 else i for i in seas])
    # Definir y retornar la primera parte del nombre del archivo
    return f'{v}_mme_{init_month_abbr}{str(init_year)}_{season}'


def hindcast_output_filename_first_section(v, ic, lt):
    # Defino ref dataset y target season
    seas = range(ic + lt, ic + lt + 3)
    season = "".join(calendar.month_abbr[i][0] for i in [i - 12 if i > 12 else i for i in seas])
    month = calendar.month_abbr[ic]
    # Definir y retornar la primera parte del nombre del archivo
    return f'{v}_mme_{month}_{season}'


def main(main_args: argparse.Namespace):

    if main_args.file_type == 'realtime_forecasts':
        desc_file_mame = 'realtime_forecasts_descriptors.yaml'
        forecasts_folder = Path(cfg.get('folders').get('gen_data_folder'),
                                cfg.get('folders').get('data').get('real_time_forecasts'))
    else:
        desc_file_mame = 'combined_forecasts_descriptors.yaml'
        forecasts_folder = Path(cfg.get('folders').get('gen_data_folder'),
                                cfg.get('folders').get('data').get('combined_forecasts'))

    with open(Path(forecasts_folder, desc_file_mame), 'w') as fp_desc:
        fp_desc.write('files:\n')

        for v in main_args.variables:
            for ic in main_args.ic_dates if main_args.file_type == 'realtime_forecasts' else main_args.ic_months:
                for lt in main_args.leadtimes:

                    if main_args.file_type == 'realtime_forecasts':
                        first_part, last_part = realtime_output_filename_first_section(v, ic, lt), ".npz"
                    else:
                        first_part, last_part = hindcast_output_filename_first_section(v, ic, lt), "_hind.npz"

                    for i in main_args.combination:
                        for j in main_args.weighting:
                            archivo = Path(forecasts_folder, f'{first_part}_gp_01_{j}_{i}{last_part}')
                            write_file_desc(fp_desc, 'ereg_prob_output', archivo)
                            if i == 'wsereg' and cfg.get('gen_det_data', False):
                                archivo_det = Path(forecasts_folder, 'determin_' + os.path.basename(archivo))
                                write_file_desc(fp_desc, 'ereg_det_output', archivo_det)

                    if main_args.file_type == 'realtime_forecasts':
                        archivo = Path(forecasts_folder, f'{first_part}_gp_01_same_count_hind.npz')
                        write_file_desc(fp_desc, 'ereg_prob_output', archivo)


# ==================================================================================================
if __name__ == "__main__":

    # Define parser data
    parser = argparse.ArgumentParser(description='Create descriptors for forecasts')
    parser.add_argument('--variables', nargs='+',
                        default=["tref", "prec"], choices=["tref", "prefc"],
                        help='Variables that was considered in the forecast generation process.')
    subparsers = parser.add_subparsers(dest='file_type')
    subparsers.required = True
    parser_real_time = subparsers.add_parser('realtime_forecasts', help='realtime forecasts help')
    parser_real_time.add_argument('--ic-dates', type=str, nargs='+', dest='ic_dates',
                                  help='Dates of initial conditions (in "YYYY-MM-DD")',
                                  required=True)
    parser_comb_fcst = subparsers.add_parser('hindcast_forecasts', help='hindcast/combined forecasts help')
    parser_comb_fcst.add_argument('--ic-months', type=int, nargs='+', dest='ic_months',
                                  default=range(1, 12+1), choices=range(1, 12+1),
                                  help='Months of initial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('--leadtimes', type=int, nargs='+',
                        default=range(1, 7+1), choices=range(1, 7+1),
                        help='Forecast leadtimes (in months, from 1 to 7)')
    parser.add_argument('--weighting', nargs='+',
                        default=["same", "pdf_int", "mean_cor"], choices=["same", "pdf_int", "mean_cor"],
                        help='Weighting methods to be plotted.')
    parser.add_argument('--combination', nargs='+',
                        default=["wpdf", "wsereg"], choices=["wpdf", "wsereg"],
                        help='Combination methods to be plotted.')

    # Extract data from args
    args = parser.parse_args()

    # Run plotting
    start = time.time()
    try:
        main(args)
    except Exception as e:
        error_detected = True
        cfg.logger.error(f"Failed to run \"create_output_files_descriptors.py\". Error: {e}.")
        raise  # see: http://www.markbetz.net/2014/04/30/re-raising-exceptions-in-python/
    else:
        error_detected = False
    finally:
        end = time.time()
        err_pfx = "with" if error_detected else "without"
        message = f"Total time to run \"create_output_files_descriptors.py\" ({err_pfx} errors): {end - start}"
        print(message) if not cfg.get('use_logger') else cfg.logger.info(message)
