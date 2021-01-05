import yaml

class Struct:
    def __init__(self, entries):
        self.__dict__.update(entries)

def Param(path):
    with open(path+"params.yml") as f:
        params = yaml.full_load(f)
    return Struct(params)
