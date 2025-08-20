import os
import subprocess
import webbrowser
from global_dir import root_dir
from torch.utils.tensorboard import SummaryWriter

# Starts TensorBoard server (one-time)
def SetupBoard(env_name, alg_name, port=6009, open_local = False):
    log_dir = os.path.join(root_dir, "tensorboard")

    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port} --bind_all"
    subprocess.Popen(tensorboard_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if open_local:
        webbrowser.open(f"http://localhost:{port}")

# Create a writer for a specific algorithm (can be called multiple times)
def get_log_dir(env_name, rl_alg_name, alter_plot_name=None):
    if alter_plot_name is not None:
        log_dir = os.path.join(root_dir, "tensorboard", env_name, rl_alg_name, alter_plot_name)
    else:
        log_dir = os.path.join(root_dir, "tensorboard", env_name, rl_alg_name)

    return log_dir
    return SummaryWriter(log_dir=log_dir, comment=f"_{rl_alg_name}")
def create_writer(env_name, rl_alg_name, alter_plot_name=None):
    log_dir = get_log_dir(env_name, rl_alg_name, alter_plot_name)
    return SummaryWriter(log_dir=log_dir, comment=f"_{rl_alg_name}")

