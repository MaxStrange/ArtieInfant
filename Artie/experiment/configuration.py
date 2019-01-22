"""
This file contains the code to load configuration files for a given experiment.
"""
import configparser
import os

class ConfigError(Exception):
    """
    An exception raised in the case of a problem with the configuration.
    """
    pass

class Configuration:
    """
    Exposes the raw configparser object via self.rawconfig, but also provides convenience
    methods that will help to parse the file and return more useful error messages to the
    user if things go wrong.
    """
    def __init__(self, rawconfigparser):
        self.rawconfig = rawconfigparser

    def _sanity_check_args(self, section, value):
        """Throws an exception if section or value aren't in the config file."""
        if not section in self.rawconfig:
            raise ConfigError("Could not find section", section, "in config file. Available sections are:", self.rawconfig.sections())
        if not value in self.rawconfig[section]:
            raise ConfigError("Could not find value {} in section {}. Available values are: {}".format(value, section, [k for k in self.rawconfig[section]]))

    def getint(self, section, value):
        """Attempts to get the value from section as an int."""
        self._sanity_check_args(section, value)
        try:
            return int(self.rawconfig[section][value])
        except ValueError as e:
            raise ConfigError("Could not convert {} to int. Error message: {}".format(self.rawconfig[section][value], e))

    def getfloat(self, section, value):
        """Attempts to get the value from section as a float."""
        self._sanity_check_args(section, value)
        try:
            return float(self.rawconfig[section][value])
        except ValueError as e:
            raise ConfigError("Could not convert {} to float. Error message: {}".format(self.rawconfig[section][value], e))

    def getlist(self, section, value, type=None):
        """
        Attempts to parse value from section as a list of items. If type is not None, will try to convert
        each item to the given type (which must be a function - like int, float, or list).
        """
        self._sanity_check_args(section, value)
        configval = self.rawconfig[section][value].strip()
        ret = []
        for item in configval.split(' '):
            if not item:
                continue
            if type is not None:
                try:
                    item = item.strip('[](),')
                    item = type(item)
                except ValueError as e:
                    msg = "Cannot convert {}:{} to a list because we could not convert {} to {}. Original error message: {}".format(
                        section, value, item, type, e
                    )
                    raise ConfigError(msg)
            ret.append(item)
        return ret

    def getstr(self, section, value):
        """Attempts to get the value from section as a str."""
        self._sanity_check_args(section, value)
        return self.rawconfig[section][value].strip()

def load(experiment_name, fpath=None):
    """
    Loads the given `fpath` if not None. If None, loads the configuration
    named '<experiment_name>.cfg' in experiment/configfiles.

    If we can't find the file, we throw a ValueError.

    :param experiment_name: The name of the experiment. Will be used as part of the file name
                            if fpath is not specified.
    :param fpath:           If specified, experiment_name is ignored and we try to load the given
                            file into a configuration.
    :returns:               Configuration instance.
    """
    if fpath is None:
        ourdir = os.path.dirname(os.path.abspath(__file__))
        fname = "{}.cfg".format(experiment_name)
        fpath = os.path.join(ourdir, "configfiles", fname)

    if not os.path.isfile(fpath):
        raise ValueError("Could not find {} to load as a config file.".format(fpath))

    config = configparser.ConfigParser()
    config.read(fpath)
    return Configuration(config)
