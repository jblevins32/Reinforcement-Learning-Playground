import numpy as np
import mujoco as mj

def DomainRandomize(env, alter_gravity, alter_friction):

    # Modify all the envs in a synced vector of envs
    floor_geom_id = 0 # This is the floor geometry id for Ant-v5 env
    num_envs = len(env.envs)

    for idx, _ in enumerate(env.envs):

        # Change the floor geometry to be bumpy
        # env.envs[idx].unwrapped.model.geom_pos[floor_geom_id] = np.ones(3)*-100000

        # Change the friction
        env.envs[idx].unwrapped.model.geom_friction[:] *= alter_friction

        # # Change the gravity
        env.envs[idx].unwrapped.model.opt.gravity[:] *= alter_gravity

        # # Change the mass 
        # env.envs[idx].unwrapped.model.body_mass[:] *= 1.1

        # Set changes

    return env

def Randomdisturbs(env, disturb_limit):

    body_name = "torso"
    body_id = mj.mj_name2id(env.envs[0].unwrapped.model, mj.mjtObj.mjOBJ_BODY, body_name) # This gets the mujoco ID of the body chosen above
    
    for idx, _ in enumerate(env.envs):

        random_force = np.random.uniform(-disturb_limit, disturb_limit, size=6)
        env.envs[idx].unwrapped.data.xfrc_applied[body_id, :] = random_force # first 3 cols of each body are forces and the last 3 are torques

    return env