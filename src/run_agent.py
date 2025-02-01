import mujoco
from mujoco import viewer
from humanoid.train import *
from humanoid.test import *
# import future agent types here

# Load and run the agent
def RunAgent(operation,t_steps,model_dir,sim=False):
    model = mujoco.MjModel.from_xml_path(model_dir)
    data = mujoco.MjData(model)

    # Train or test with the Mujoco simulation
    if sim == True:
        with viewer.launch_passive(model, data) as v:\
            
            # Running for t timesteps
            for _ in range(t_steps):

                # Train or Test the agent
                if operation == "1":
                    Train(data)
                else:
                    Test()

                # Step Mujoco
                mujoco.mj_step(model, data)
                v.sync()

    # Train or test without the Mujoco simulation
    else:
        # Running for t timesteps
        for _ in range(t_steps):

            # Train or Test the agent
            if operation == "1":
                Train(data)
            else:
                Test()

            # Step Mujoco
            mujoco.mj_step(model, data)
