from humanoid_sim import run_agent
import multiprocessing
import time

operation = input("Select 1 for Training or 2 for Testing: ")

# Create variables and parallel processes
num_environments = 30
t_steps=100000
sim=False

processes = [multiprocessing.Process(target=run_agent,args=(operation,t_steps,num_environments,sim)) for _ in range(num_environments)]

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