from run_agent import RunAgent
import multiprocessing
import time
from globals import root_directory
import os
import yaml

operation = input("Select 1 for Training or 2 for Testing: ")

# Import args
config_path = os.path.join(root_directory, "config.yaml")
with open(config_path, "r") as read_file:
    config = yaml.safe_load(read_file)

# Create parallel processes for all environments
processes = [multiprocessing.Process(target=RunAgent,args=(operation,config['t_steps'],config['model_dir'],config['sim'])) for _ in range(config['num_environments'])]

# For tracking sim times
time_tracker = {}

# Main run
for idx, process in enumerate(processes):
    time_tracker[idx] = time.time()
    process.start()

for idx, process in enumerate(processes):
    print(f"Environment {idx+1} took {time.time()-time_tracker[idx]} seconds.")
    process.join()

print("Done!")