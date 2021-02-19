import yaml
import argparse

class Struct:
    def __init__(self, entries):
        self.__dict__.update(entries)

def Param(path):
    with open(path+"params.yml") as f:
        params = yaml.full_load(f)
    return Struct(params)

def arg(train=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="config path", type=str)
    if not train: parser.add_argument("epoch", help="epoch", type=int)
    return parser.parse_args()
