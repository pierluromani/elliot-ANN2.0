from elliot.run import run_experiment
import subprocess
import sys

"""
SCRIPT TO PREPROCESS THE MOVIELENS100K FOR THRESHOLD EXPERIMENT

BEFORE RUNNING THIS SCRIPT, DELETE THE /data/movielens_100k/filterd_data 
folder 
"""

print("We are now starting the data preprocessing with Elliot")
run_experiment("config_preprocess_dataset/preprocess_movielens_100k.yml")


# Define the command and its arguments as a list
command = [
    sys.executable,        # Use the CURRENT interpreter (elliot_venv)
    "split_into_3_groups_tolerance.py",           # The file you want to run
    "--dataset",           # Argument 1
    "movielens_100k"       # Argument 2
]

print(f"Running command: {' '.join(command)}")

# Execute the command
result = subprocess.run(command, capture_output=True, text=True)

# Check if it worked
if result.returncode == 0:
    print("Success!")
    print(result.stdout)
else:
    print("Error occurred:")
    print(result.stderr)