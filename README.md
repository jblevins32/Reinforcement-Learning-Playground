# Reinforcement Learning Playground
This repo holds my Reinforcement Learning algorithms which are run in gym environments

## File Structure
- `data`: Git ignored folder for storing reward data
- `models`: Git ignored folder for storing trained models
- `src`: Source code
    - `incomplete_RL`: Incomplete RL algorithms
    - `my_sim`: My custom Gym environment for Multi Robot Path Planning
    - `RL_algorithms`: Completed RL algorithms
    - `agent.py`: Main RL agent code, runs epochs, rollouts, and updates
    - `buffer.py`: Buffer class for storing data from rollouts
    - `create_env.py`: Calls for creation of the environment with specific indices for each env type
    - `get_params.py`: Loads data from `config.yaml`
    - `globals.py`: Creates global directory
    - `tensorboard_setup.py`: Instantiates tesnsorboard
    - `train.py`: RUN THIS FOR TRAINING
    - `train_adv.py`: RUN THIS FOR TRAINING ADVERSARIALLY
    - `test.py`: RUN THIS FOR TESTING
- `tensorboard`: Git ignored folder for storing tensorboard data
- `videos`: Git ignored folder for storing test videos
- `changes.md`: Future work
- `config.yaml`: Parameters to adjust for training and inference
- `environment.yaml`: Environment Setup. NOT CORRECT RIGHT NOW
- `README.md`: This is literally you right now...
- `RL_REQS.md`: RL job listings to drool over

