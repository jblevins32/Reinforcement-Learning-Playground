import torch

def RobotModel(old_state, control):
    
    # Model = integrator
    # Control = x and y velocities
    # State = x and y position
    control_gain = 0.5
    state = old_state + control_gain*control
    
    return torch.tensor(state,dtype=torch.float32)