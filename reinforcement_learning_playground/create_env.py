import gymnasium as gym
from get_params import GetParams
from tensorboard_setup import *

def CreateEnv(operation):

    # Import args from config.yaml
    config = GetParams()

    # Tensor board setup
    SetupBoard()
    writer = create_writer(config['rl_alg_name'])

    # Create environment
    if operation == "train":
        if config['gym_model_train'] == "MRPP_Env":
            env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model_train'], render_mode="rgb_array", **config) for _ in range(config['num_environments'])])
            n_actions = env.action_space.shape[1]*env.action_space.shape[2]
            n_obs = env.observation_space.shape[1]*env.observation_space.shape[2]
        else:
            if config['space'] == "cont":
                env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model_train'], render_mode="rgb_array") for _ in range(config['num_environments'])])
                n_actions = env.action_space.shape[1]
                n_obs = env.observation_space.shape[1]
            elif config['space'] == "disc":
                env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model_train'], render_mode="rgb_array") for _ in range(config['num_environments'])])
                n_actions = 2 # 2 for cart-pole
                n_obs = env.observation_space.shape[1]
            
    elif operation == "test":
        if config['gym_model_test'] == "MRPP_Env":
            env = gym.make(config['gym_model_test'], render_mode="rgb_array", **config)
            n_actions = env.action_space.shape[0]*env.action_space.shape[1]
            n_obs = env.observation_space.shape[0]*env.observation_space.shape[1]
        else:
            env = gym.make(config['gym_model_test'], render_mode="rgb_array")
            n_actions = env.action_space.shape[0]
            n_obs = env.observation_space.shape[0]

    return env, n_obs, n_actions, writer, config