import gymnasium
import numpy as np

xml = '/home/jblevins32/.cache/robot_descriptions/mujoco_menagerie/unitree_go2/scene.xml'
env = gymnasium.make(
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
    max_episode_steps=1000,
)

