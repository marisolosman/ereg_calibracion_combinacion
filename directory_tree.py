"""This code creates all the directories needed to run Ensemble regression
Input: Main directory where files will be saved
"""
import os
import sys
import pathlib

def main():
    """Create directories to storage data"""
    TREE = sys.argv[1]
    pathlib.Path(TREE).mkdir(parents=True, exist_ok=True)
    #NMME: hindcast y real_time
    pathlib.Path(os.path.join(TREE, 'NMME', 'hindcast')).mkdir(parents=True,
                                                               exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'NMME', 'real_time')).mkdir(parents=True,
                                                                exist_ok=True)
    #DATA: Observations, calibrated_forecast, combined_forecast, real_time_forecast
    pathlib.Path(os.path.join(TREE, 'DATA', 'Observations')).mkdir(parents=True,
                                                                   exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'DATA', 'calibrated_forecasts')).mkdir(parents=True,
                                                                           exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'DATA', 'combined_forecasts')).mkdir(parents=True,
                                                                         exist_ok=True)
    pathlib.Path(os.path.join(TREE, 'DATA', 'real_time_forecasts')).mkdir(parents=True,
                                                                          exist_ok=True)
    #FIGURES
    pathlib.Path(os.path.join(TREE, 'FIGURES')).mkdir(parents=True, exist_ok=True)
    print("Directories created")
    file1 = open("configuracion", "w")
    file1.write(TREE)
    file1.close()

#================================================================================================
if __name__ == "__main__":
    main()


