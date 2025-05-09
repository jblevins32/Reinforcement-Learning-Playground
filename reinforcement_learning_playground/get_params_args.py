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
    parser.add_argument('--open_local', action='store_true',  default=False, help="Open Tensorboard before training")
    parser.add_argument('--render', action='store_true',  default=False, help="Open MuJoCo sim for viewing training in one env in real time")
    parser.add_argument('--alter_gravity', type=float, default=None, help="Set a gravity value for domain randomization")
    parser.add_argument('--alter_friction', type=float, default=None, help="Set a friction value for domain randomization")
    parser.add_argument('--disturb_limit', type=float, default=None, help="Set a random disturb force limit for domain randomization")
    parser.add_argument('--disturb_rate', type=float, default=None, help="Set a random disturb rate for domain randomization. 1 = 100%, 0 = 0%")
    parser.add_argument('--alter_plot_name', type=str, default=None, help="Set a unique name for this plot in tensorboard")
    parser.add_argument('--model', type=str, default=None, help="Set a model to use for testing")

    args = parser.parse_args()

    return args