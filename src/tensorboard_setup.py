import os
from globals import root_dir
from torch.utils.tensorboard import SummaryWriter
import webbrowser
import subprocess

def SetupBoard(rl_alg_name):
    log_dir=os.path.join(root_dir,"tensorboard",rl_alg_name)

    # Start the tensorboard
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port=6006 --bind_all"
    subprocess.Popen(tensorboard_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    webbrowser.open("http://localhost:6006")

    # Create the writer
    return SummaryWriter(log_dir=log_dir, comment=f"_{rl_alg_name}")
