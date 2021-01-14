
import singleton
import yaml
import glob
import datetime
import os
import logging.config
import logging
import sys
import pathlib


@singleton.Singleton
class Config():
    """Class that manage the configuration file.  """

    def __init__(self, config = 'config.yaml', logger_config = 'logging.yaml', email_config = 'email.yaml'):
        self.file = config
        self.config = self._load_ereg_config(config)
        self.logger = self._load_logging_config(logger_config)
        self.email = self._load_email_config(email_config)
        self._update_models()
        self._check_models()
        self._setup_directory_tree()

    def get(self, keyname):
        if keyname not in self.config:
            raise InvalidConfiguration(f"Yaml file \"{self.file}\" don't contain this entry: {keyname}")
        return self.config.get(keyname)
    
    def _load_ereg_config(self, yaml_file):
        if not os.path.exists(yaml_file):
            raise InvalidConfiguration(f"Configuration file (i.e. {yaml_file}) not found!")
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)
                                   
    def _load_logging_config(self, yaml_file, default_level=logging.INFO):
        logger, message = None, None
        if os.path.exists(yaml_file):
            with open(yaml_file, 'rt') as f:
                try:
                    config = yaml.safe_load(f.read())
                    logging.config.dictConfig(config)
                except Exception as e:
                    logging.basicConfig(level=default_level)
                    message = f"Error loading logging configuration file. Using default configs!"
        else:
            logging.basicConfig(level=default_level)
            message = f"Logging configuration file (i.e. {yaml_file}) not found!. Using default configs!"
        logger = logging.getLogger('ereg')
        logger.warning(message) if message else None
        return logger
    
    def _load_email_config(self, yaml_file):
        email_config = None
        if os.path.exists(yaml_file):
            with open(yaml_file, 'rt') as f:
                try:
                    email_config = yaml.safe_load(f.read())
                except Exception as e:
                    self.logger.error(f"Failed to load email configuration. File {yaml_file} couldn't be read!")
        return email_config
    
    @property
    def _is_there_comb_forecasts(self):
        gen_data_folder = self.config.get('gen_data_folder')
        PATH = f"{gen_data_folder}/nmme_output/comb_forecast/*".replace("//","/")
        return any(glob.glob(PATH))
    
    @property
    def _models_in_config(self):
        return [ item[0] for item in self.config.get('models')[1:] ]
    
    @property
    def _combined_models(self):
        with open("combined_models", "r") as f:
            return f.read().strip().split("\n")
    
    def _delete_combined_forecasts(self):
        gen_data_folder = self.config.get('gen_data_folder')
        PATH = f"{gen_data_folder}/nmme_output/comb_forecast/*".replace("//","/")
        for filename in glob.glob(PATH):
            os.remove(filename)
    
    def _update_updates_file(self):
        fecha = datetime.datetime.now().strftime('%d/%m/%Y')
        deleted_models = set(self._combined_models).difference(self._models_in_config)
        if deleted_models:
            with open("updates", "a") as f:
                for model in deleted_models:
                    f.write(f"{fecha}: Model {model} discarded" + "\n")
        added_models = set(self._models_in_config).difference(self._combined_models)
        if added_models:
            with open("updates", "a") as f:
                for model in added_models:
                    f.write(f"{fecha}: Model {model} added" + "\n")
    
    def _update_combined_models_file(self):
        with open("combined_models", "w") as f:
            for model in self._models_in_config:
                f.write(model + "\n")
    
    def _update_models(self):
        if set(self._combined_models).symmetric_difference(self._models_in_config):
            r = input("Model/s was added or deleted. Do you want to drop current "+
                      "combined forecasts and update combined_models file? (y/n): ")
            if r == 'y' or r == 'Y':
                self._delete_combined_forecasts()
                self._update_updates_file()
                self._update_combined_models_file()
        
    def _check_models(self):
        deleted_models = set(self._combined_models).difference(self._models_in_config)
        if deleted_models and self._is_there_comb_forecasts:
            raise InvalidConfiguration(f"Model/s \"{', '.join(deleted_models)}\" was "+
                                       f"deleted, but combined forecasts wasn't dropped.")
        added_models = set(self._models_in_config).difference(self._combined_models)
        if added_models and self._is_there_comb_forecasts:
            raise InvalidConfiguration(f"Model/s \"{', '.join(added_models)}\" was "+
                                       f"added, but combined forecasts wasn't dropped.")
        models_data = [m[0] for m in self.get('models')[1:]]
        models_urls = [m[0] for m in self.get('models_url')[1:]]
        if models_data != models_urls:
            raise InvalidConfiguration(f"Model/s in \"model\" tag en model/s in \"model_url\" tag mismatch. "+
                                       f"Please correct file \"{self.file}\" and try again.")
        
    def _setup_directory_tree(self):
        """Create directories to storage data (if needed)"""

        download_folder = self.get('download_folder')
        if not os.access(pathlib.Path(download_folder).parent, os.W_OK):
            sys.exit(f"{pathlib.Path(download_folder).parent} is not writable")
        pathlib.Path(download_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(download_folder, 'NMME', 'hindcast')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(download_folder, 'NMME', 'real_time')).mkdir(parents=True, exist_ok=True)

        gen_data_folder = self.get('gen_data_folder')
        if not os.access(pathlib.Path(gen_data_folder).parent, os.W_OK):
            sys.exit(f"{pathlib.Path(gen_data_folder).parent} is not writable")
        pathlib.Path(gen_data_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(gen_data_folder, 'nmme_output', 'cal_forecasts')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(gen_data_folder, 'nmme_output', 'comb_forecast')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(gen_data_folder, 'nmme_output', 'rt_forecast')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(gen_data_folder, 'nmme_figuras', 'forecast')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(gen_data_folder, 'nmme_figuras', 'rt_forecast')).mkdir(parents=True, exist_ok=True)
            


class InvalidConfiguration(Exception):
    """Exception raised when an invalid configuration was detected. """

    def __init__(self, message):
        self.message = message
