import numpy as np

def DomainRandomize(env, alter_gravity):

    # Modify all the envs in a synced vector of envs
    floor_geom_id = 0 # This is the floor geometry id for Ant-v5 env
    num_envs = len(env.envs)

    for idx, _ in enumerate(env.envs):

        # Change the floor geometry to be bumpy
        # env.envs[idx].unwrapped.model.geom_pos[floor_geom_id] = np.ones(3)*-100000

        # Change the friction
        # env.envs[idx].unwrapped.model.geom_friction[:] *= 1.1

        # # Change the gravity
        env.envs[idx].unwrapped.model.opt.gravity[:] *= alter_gravity

        # # Change the mass 
        # env.envs[idx].unwrapped.model.body_mass[:] *= 1.1

        # Set changes

    return env
