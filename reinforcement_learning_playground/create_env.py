import gymnasium as gym
from get_params import GetParams
from tensorboard_setup import *

def CreateEnv(operation, open_local = False):

    # Import args from config.yaml
    config = GetParams()

    # Tensor board setup
    SetupBoard(open_local=open_local)
    writer = create_writer(config['rl_alg_name'])

    # Create environment
    if config['operation'] == "train":
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
            
    elif config['operation'] == "test":
        if config['gym_model_test'] == "MRPP_Env":
            env = gym.make(config['gym_model_test'], render_mode="rgb_array", **config)
            n_actions = env.action_space.shape[0]*env.action_space.shape[1]
            n_obs = env.observation_space.shape[0]*env.observation_space.shape[1]
        else:
            env = gym.make(config['gym_model_test'], render_mode="rgb_array")
            n_actions = env.action_space.shape[0]
            n_obs = env.observation_space.shape[0]

    elif config['operation'] == "test_quad":
        xml = '/home/jblevins32/.cache/robot_descriptions/mujoco_menagerie/unitree_go1/scene.xml'
        env = gym.make(
            'Ant-v5', 
            xml,
            forward_reward_weight=1,
            ctrl_cost_weight=.05,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75), # Avoid sampling when robot has fallen
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
        )
        n_actions = env.action_space.shape[0]
        n_obs = env.observation_space.shape[0]

    return env, n_obs, n_actions, writer, config