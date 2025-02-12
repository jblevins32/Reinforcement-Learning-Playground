from globals import root_dir
import os
import yaml

def GetParams():
    config_dir = os.path.join(root_dir, "config.yaml")
    with open(config_dir, "r") as read_file:
        config = yaml.safe_load(read_file)

    return config