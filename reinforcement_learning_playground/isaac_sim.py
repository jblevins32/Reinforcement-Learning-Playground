from isaacgym import gymapi
from isaacgym import gymtorch

sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.physx.use_gpu = True

# Enable GPU pipeline
sim_params.use_gpu_pipeline = True

# create new simulation
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params)

# Prepare assets... blueprints for objects/robots to use later
quad_asset = gym.load_aset(sim,"")

# Create envs and add actors
for i in range(num_envs):
    env = gym.create_env(sim,...)

    quad = gym.create_actor(env, quad_asset, quad_pose,...)

    # Configure initial poses, etc

# Prepare simulation buffers and tensor storage - required to use tensor API
gym.prepare_sim(sim)