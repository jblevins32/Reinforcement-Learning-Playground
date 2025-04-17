from global_dir import root_dir
import os
import yaml
import argparse

def GetParams():

    # Load parameters from config.yaml
    config_dir = os.path.join(root_dir, "config.yaml")
    with open(config_dir, "r") as read_file:
        config = yaml.safe_load(read_file)

    return config

def GetArgs():
    # Gather arguments from terminal and replace parameters with that
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_alg', type=str, default=None, help="Replace the config RL alg with your choice")
    parser.add_argument('--open_local', action='store_true', help="Open Tensorboard before training")
    parser.set_defaults(open_local=False)

    args = parser.parse_args()

    return args