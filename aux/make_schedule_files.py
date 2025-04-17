import yaml
from toast.schedule_sim_ground import run_scheduler

# Load parameters from YAML file
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

run_scheduler(opts=params)

print("Wrote schedule file!")
