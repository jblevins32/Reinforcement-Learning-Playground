# Reinforcement Learning Playground
This repo holds my Reinforcement Learning algorithms which are run in gym environments

## File Structure
- `data`: Git ignored folder for storing reward data
- `models`: Git ignored folder for storing trained models
- `src`: Source code
    - `incomplete_RL`: Incomplete RL algorithms
    - `my_sim`
        - `gym_simulation.py`: My custom Gym environment for Multi Robot Path Planning
        - `test_my_sim.py`: Run tests on my custom sim either with regular controllers or RL
    - `research`:
        - `attack.py`: Adversarial attacks
        - `train_adv.py`: Robust Adversarial RL training
    - `RL_algorithms`: Completed RL algorithms
    - `agent.py`: Main RL agent code, runs epochs, rollouts, and updates
    - `buffer.py`: Buffer class for storing data from rollouts
    - `create_env.py`: Calls for creation of the environment with specific indices for each env type
    - `get_params.py`: Loads data from `config.yaml`
    - `global_dir.py`: Creates global directory
    - `tensorboard_setup.py`: Instantiates tesnsorboard
    - `test.py`: RUN THIS FOR TESTING
    - `train.py`: RUN THIS FOR TRAINING
- `tensorboard`: Git ignored folder for storing tensorboard data
- `videos`: Git ignored folder for storing test videos
- `changes.md`: Future work
- `config.yaml`: Parameters to adjust for training and inference
- `environment.yaml`: Environment Setup. NOT CORRECT RIGHT NOW
- `README.md`: This is literally you right now...
- `RL_REQS.md`: RL job listings to drool over

## Install environment:
- `conda env create -f environment.yaml`
- `python -m pip install .`

