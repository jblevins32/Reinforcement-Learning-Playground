import gymnasium as gym
from get_params_args import *
from tensorboard_setup import *
from gymnasium.spaces import Box
from domain_rand import DomainRandomize

def CreateEnv(operation):

    # Import args from flags and parameters from config.yaml
    args = GetArgs()
    config = GetParams(args)

    # For testing on env specifically in chosen test model
    # gym_model = config['test_model'].split('_')[0]
    gym_model = config['gym_model_test']

    # Create environment
    if operation == "train":

        # Tensor board setup
        SetupBoard(config['gym_model'], config['rl_alg_name'],open_local=args.open_local)
        writer = create_writer(config['gym_model'],config['rl_alg_name'], config["alter_plot_name"])

        # Isolate the first env for human rendering in mujoco if desired
        envs = []
        for idx in range(config['num_environments']):
            if idx == 0:
                envs.append(lambda: gym.make(config['gym_model'], render_mode=config["render_mode_env_zero"]))
            else:
                envs.append(lambda: gym.make(config['gym_model'], render_mode="rgb_array"))

        env = gym.vector.SyncVectorEnv(envs)

        # env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model'], render_mode="rgb_array") for _ in range(config['num_environments'])])
                
        # Define the space as cont or disc to get the correct actions and obs spaces
        if isinstance(env.action_space, Box):
            space = "cont"
        else: space = "disc"

        if space == "cont":
            n_actions = env.action_space.shape[1]
            n_obs = env.observation_space.shape[1]
        else:
            n_actions = 2 # 2 for cart-pole
            n_obs = env.observation_space.shape[1]

        if config['gym_model'] == "MRPP_Env":
            n_actions = env.action_space.shape[1]*env.action_space.shape[2]
            n_obs = env.observation_space.shape[1]*env.observation_space.shape[2]

        # Domain randomization, don't worry about for custom sim right now
        if config['gym_model'] != "MRPP_Env":
            env = DomainRandomize(env, config["alter_gravity"], config["alter_friction"])
            
    elif operation == "test":
        writer = None

        if gym_model == "MRPP_Env":
            env = gym.make(gym_model, render_mode="rgb_array", **config)
            n_actions = env.action_space.shape[0]*env.action_space.shape[1]
            n_obs = env.observation_space.shape[0]*env.observation_space.shape[1]
        else:
            env = gym.make(gym_model, render_mode="rgb_array")
            n_actions = env.action_space.shape[0]
            n_obs = env.observation_space.shape[0]

    elif operation == "test_quad":
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