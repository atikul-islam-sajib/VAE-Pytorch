import yaml
import joblib


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)


class CustomException(Exception):
    def __init__(self, message=None):
        super().__init__(message)
