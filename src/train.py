# Main training logic
import numpy as np

def Train(data):

    # Adjust the joint velocities
    joint_vels = np.array([0,.1,-.1,.2,-.3,.15,0,.1,-.1,.2,-.3,.15,0,.1,-.1,.2,-.3,.15,.1,.1,.1,.2,-.3,.15,.1,.1,.1])
    data.qvel = joint_vels*.1

    return data