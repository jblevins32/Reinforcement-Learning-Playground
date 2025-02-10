from run_agent import RunAgent
import multiprocessing
import time
from globals import root_directory
import os
import yaml

# Import args from config.yaml
config_path = os.path.join(root_directory, "config.yaml")
with open(config_path, "r") as read_file:
    config = yaml.safe_load(read_file)

# Create parallel processes for chosen number of environments
processes = [multiprocessing.Process(target=RunAgent,args=(config['operation'],config['num_environments'],config['epochs'],config['t_steps'],config['model_dir'],config['env'],config['live_sim'],config['discount'],config['epsilon'],config['lr'])) for _ in range(config['num_environments'])]

# For tracking sim times
time_tracker = {}

# Start all parallel processes
for idx, process in enumerate(processes):
    time_tracker[idx] = time.time()
    process.start()

# Wait for all parallel processes to finish
for idx, process in enumerate(processes):
    process.join()
    print(f"Environment {idx+1} took {time.time()-time_tracker[idx]} seconds.")

print("Done!")