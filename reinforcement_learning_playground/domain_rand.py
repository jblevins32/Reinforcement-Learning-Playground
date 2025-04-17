import numpy as np

def DomainRandomize(env):

    # Modify all the envs in a synced vector of envs
    floor_geom_id = 0 # This is the floor geometry id for Ant-v5 env
    num_envs = len(env.envs)

    for idx, _ in enumerate(env.envs):

        # Change the floor geometry to be bumpy
        env.envs[idx].unwrapped.model.geom_pos[floor_geom_id] = np.ones(3)*-100000

        # Change the friction
        # env_model.geom_friction[:] *= 1.1

        # # Change the gravity
        # env_model.opt.gravity[:] *= 1.1

        # # Change the mass 
        # env_model.body_mass[:] *= 1.1

        # Set changes
        

    return env
