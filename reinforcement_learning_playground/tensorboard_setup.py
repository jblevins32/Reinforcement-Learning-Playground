import os
import subprocess
import webbrowser
from global_dir import root_dir
from torch.utils.tensorboard import SummaryWriter

# Starts TensorBoard server (one-time)
def SetupBoard(port=6009, open_local = False):
    log_dir = os.path.join(root_dir, "tensorboard")

    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port} --bind_all"
    subprocess.Popen(tensorboard_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if open_local:
        webbrowser.open(f"http://localhost:{port}")

# Create a writer for a specific algorithm (can be called multiple times)
def create_writer(rl_alg_name):
    log_dir = os.path.join(root_dir, "tensorboard", rl_alg_name)
    return SummaryWriter(log_dir=log_dir, comment=f"_{rl_alg_name}")
