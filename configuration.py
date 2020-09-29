import singleton
import yaml


def load_yaml(filename):
    with open(filename, 'r') as yamlfile:
        loaded_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return loaded_yaml


@singleton.Singleton
class Config():
    """Class that manage the configuration file.  """

    def __init__(self, filename = 'config.yaml'):
        self.file = filename
        self.yaml = load_yaml(filename)

    def get(self, keyname):
        if keyname not in self.yaml:
            raise InvalidConfiguration(f"{self.file} must contains this entry: {keyname}")
        return self.yaml.get(keyname)


class InvalidConfiguration(Exception):
    """Exception raised when an invalid configuration was detected. """

    def __init__(self, message):
        self.message = message
