import builtins
if not hasattr(builtins, 'long'):
    builtins.long = int

from elliot.run import run_experiment
from elliot.utils.custom_logging import create_CR_dataset
import yaml
import os

config_path="config_stat_test/target_t_0_05.yaml"
# excluded_params = ["meta", "path_model"]

# create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)
config_path="config_stat_test/target_t_0_25.yaml"
# excluded_params = ["meta", "path_model"]

# create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)