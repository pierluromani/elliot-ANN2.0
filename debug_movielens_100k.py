import builtins
if not hasattr(builtins, 'long'):
    builtins.long = int

from elliot.run import run_experiment
from elliot.utils.custom_logging import create_CR_dataset
import yaml
import os

config_path="config_target_threshold/prova.yaml"
excluded_params = ["meta", "path_model"]

create_CR_dataset(config_path, excluded_params)

run_experiment(config_path)