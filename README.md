# Reinforcement Learning Playground
This repo holds my Reinforcement Learning algorithms and research which are run in OpenAI's gymnasium environments

## File Structure
- `data`: Storage for reward/loss data from training
- `height_field`: Files and instructions for modified MuJoCo Ant-v5 with hfield
- `models`: Git ignored folder for storing trained models
- `models_best`: Best models to keep
- `MuJoCo`: My messing around with raw MuJoCo and Isaac Sim. Ignore.
- `reinforcement_learning_playground`: Source code
    - `incomplete_RL`: Incomplete RL algorithms
    - `my_sim`
        - `gym_simulation.py`: My custom Gym environment for Multi Robot Path Planning
        - `test_my_sim.py`: Run tests on my custom sim
    - `research`:
        - `attack.py`: Adversarial attacks
        - `smsf.py`: State monitoring function for detecting attacks
        - `train_adv.py`: Robust Adversarial RL training
    - `RL_algorithms`: Completed RL algorithms
    - `agent.py`: Main RL agent code, runs epochs, rollouts, and updates
    - `create_env.py`: Calls for creation of the environment with specific indices for each env type
    - `domain_rand.py`: Domain randomization function
    - `get_action.py`: General function for getting actions from policies
    - `get_params_args.py`: Loads data from `config.yaml` and terminal args
    - `global_dir.py`: Creates global directory
    - `replay_buffer.py`: Storing data from rollouts for off-policy algorithms
    - `tensorboard_setup.py`: Instantiates tesnsorboard
    - `test_sb.py`: Rough test for stablebaselines3
    - `test.py`: RUN THIS FOR TESTING
    - `train_multiple.sh`: Shell script for training/testing in batches
    - `train.py`: RUN THIS FOR TRAINING
    - `traj_data.py`: Storing data from rollouts for on-policy algorithms
- `tensorboard`: Git ignored folder for storing tensorboard data
- `videos`: Git ignored folder for storing test videos
- `changes.md`: Future work
- `config.yaml`: Parameters to adjust for training and inference
- `environment.yaml`: Environment Setup. Not verified.
- `generate_requirements.py`: requirements.txt generator. Does not work right now.
- some poetry stuff
- `README.md`: This is literally you right now...
- `RL_REQS.md`: RL job listings to drool over (please don't take them from me)

## Install environment:
- `conda env create -f environment.yaml`
- `python -m pip install .`

## Train Model:
- Set parameters of choice in `config.yaml` under REGULAR TRAINING/TESTING training specific parameters
- `python3 train.py` with possible arguments:
    - `--rl_alg <rl alg name>`: Run a different choice of RL algorithm from that listed in the config
    - `--open_local`: Open tensorboard, if this is not given, default is not open
    - `--render`: Render one training env, if this is not given, default is not open
    - `--alter_gravity`: Set a gravity multiplicative value for domain randomization
    - `--alter_friction`: Set a friction multiplicative value for domain randomization
    - `--disturb limit`:  Set a disturb force limit for domain randomization
    - `--disturb_rate`: Set a random disturb rate for domain randomization. 1 = 100%, 0 = 0%
    - `--alter_plot_name`: Set a unique name for this plot in tensorboard, video, and model saving

## Testing Model:
- Set parameters of choice in `config.yaml` under REGULAR TRAINING/TESTING testing specific parameters
- `python3 test.py` with possible arguments:
    - `--alter_gravity`: Set a model to use for testing
    - Rest same as training



