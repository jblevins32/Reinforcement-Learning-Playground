from global_dir import root_dir
import os
import yaml
import argparse
import warnings

def GetParams(args):

    # Load parameters from config.yaml
    config_dir = os.path.join(root_dir, "config.yaml")
    with open(config_dir, "r") as read_file:
        config = yaml.safe_load(read_file)

    ### Replace config variables if args are given to replace them
    # Replace rl_alg with argument choice if there is one
    if args.rl_alg is not None:
        config["rl_alg_name"] = args.rl_alg

    # Render the training env if called for in flag
    if args.render:
        config["render_mode_env_zero"] = "human"
    else:
        config["render_mode_env_zero"] = "rgb_array"

    # Domain randomization stuff
    if args.alter_gravity is not None:
        config["alter_gravity"] = args.alter_gravity
    else: config["alter_gravity"] = 1

    if args.alter_friction is not None:
        config["alter_friction"] = args.alter_friction
    else: config["alter_friction"] = 1

    # For naming the models with their unique env modifications
    if args.alter_plot_name is not None:
        config["alter_plot_name"] = args.alter_plot_name
    else:
        config["alter_plot_name"] = 'no-mods'

    # Add disturbs to the model for domain randomization
    if args.disturb_limit is not None:
        config["disturb_limit"] = args.disturb_limit
    else: 
        config["disturb_limit"] = 0
        if args.disturb_rate is not None:
            warnings.warn("disturb rate chosen, but no disturb limit. No disturb will be applied.")
    if args.disturb_rate is not None:
        config["disturb_rate"] = args.disturb_rate
    else:
        config["disturb_rate"] = 0
        if args.disturb_limit is not None:
            warnings.warn("disturb limit chosen, but no disturb rate. No disturb will be applied.")

    return config

def GetArgs():
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