import builtins
if not hasattr(builtins, 'long'):
    builtins.long = int

from elliot.run import run_experiment
from elliot.utils.custom_logging import create_CR_dataset
import yaml
import os

# 0.05
config_path="config_target_threshold/target_t_0_05_minh.yaml"
excluded_params = ["meta", "path_model"]

create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)

# 0.25
config_path="config_target_threshold/target_t_0_25_minh.yaml"
excluded_params = ["meta", "path_model"]

create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)

# 0.5

config_path="config_target_threshold/target_t_0_50_minh.yaml"
excluded_params = ["meta", "path_model"]

create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)

#0.75

config_path="config_target_threshold/target_t_0_75_minh.yaml"
excluded_params = ["meta", "path_model"]

create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)

#0.95
config_path="config_target_threshold/target_t_0_95_minh.yaml"
excluded_params = ["meta", "path_model"]

create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)