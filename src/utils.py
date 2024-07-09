import yaml


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


class CustomException(Exception):
    def __init__(self, message=None):
        super().__init__(message)
