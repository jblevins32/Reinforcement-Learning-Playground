# Reinforcement Learning Playground
This repo holds my Reinforcement Learning algorithms which are run in gym environments

## File Structure
- `data`: Git ignored folder for storing reward data
- `models`: Git ignored folder for storing trained models
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
    - `get_action.py`: General function for getting actions from policies
    - `get_params.py`: Loads data from `config.yaml`
    - `global_dir.py`: Creates global directory
    - `replay_buffer.py`: Storing data from rollouts for off-policy algorithms
    - `tensorboard_setup.py`: Instantiates tesnsorboard
    - `test.py`: RUN THIS FOR TESTING
    - `train.py`: RUN THIS FOR TRAINING
    - `traj_data.py`: Storing data from rollouts for on-policy algorithms
- `tensorboard`: Git ignored folder for storing tensorboard data
- `videos`: Git ignored folder for storing test videos
- `changes.md`: Future work
- `config.yaml`: Parameters to adjust for training and inference
- `environment.yaml`: Environment Setup. NOT CORRECT RIGHT NOW
- some poetry stuff
- `README.md`: This is literally you right now...
- `RL_REQS.md`: RL job listings to drool over (please don't take them from me)

## Install environment:
- `conda env create -f environment.yaml`
- `python -m pip install .`

## Train Model:
- Set parameters of choice in `config.yaml`
- `python3 train.py` with possible arguments
    - `--rl_alg <rl alg name>`: Run a different choice of RL algorithm
    - `--open_local`: Open tensorboard, if this is not present, default is not open
    - `--render_training`: Render one training env, if this is not present, default is not open


