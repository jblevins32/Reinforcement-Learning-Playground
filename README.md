# Reinforcement Learning Playground
This repo holds my Reinforcement Learning algorithms which can be run in various environments

## File Structure
- `config.yaml`: parameters to adjust for training and inference
- `run_me.py`: primary file to run. 
    - grabs parameters from config
    - instantiates multiprocesses for multithreading
    - calls the agent
- `agent.py`: sets up environment and starts training or testing
    - chooses environment/simulator. Current options are the custom 2D env or MuJoCo
    - chooses RL algorithm to run
    - runs training or testing with/without live simulation
- `train.py`: main training script, agnostic to environment and rl algorithm 
    - updates the environment and agent according to the chosen RL algorithm until training is done
- `test.py`: main inference script, agnostic to environment and rl algorithm 
    - updates the environment and agent according to the chosen RL algorithm until a goal is reached
