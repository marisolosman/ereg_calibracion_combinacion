"""This code creates all the directories needed to run Ensemble regression. """
import os
import sys
import pathlib
import yaml
import configuration

cfg = configuration.Config.Instance()

def main():
    """Create directories to storage data"""

    TREE = cfg.get('download_folder')
    if not os.access(pathlib.Path(TREE).parent, os.W_OK):
        sys.exit(f"{pathlib.Path(TREE).parent} is not writable")

    pathlib.Path(TREE).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'NMME', 'hindcast')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'NMME', 'real_time')).mkdir(parents=True, exist_ok=True)

    TREE = cfg.get('gen_data_folder')
    if not os.access(pathlib.Path(TREE).parent, os.W_OK):
        sys.exit(f"{pathlib.Path(TREE).parent} is not writable")

    pathlib.Path(TREE).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'nmme', 'monthly', 'real_time')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'nmme_output', 'cal_forecasts')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'nmme_output', 'comb_forecast')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'nmme_output', 'rt_forecast')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'nmme_figuras', 'forecast')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'nmme_figuras', 'rt_forecast')).mkdir(parents=True, exist_ok=True)

    print("Directories created")


# ==================================================================================================
if __name__ == "__main__":
    main()

