from elliot.run import run_experiment
import yaml
import os
import pandas as pd

config_path="config_run_experiments/Debug_item_experiment_movielens_100k.yml"
exluded_params = ["meta", "path_model"]

def create_CR_dataset(config_path:str, exluded_params:list):
    # Parse the configuration file 
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)

    # Create a list of all the parameters to be recorded
    records=["CR","Model"]
    
    # Iterate all the models and collect their parameters
    for model_name, config in data["experiment"]["models"].items():
        parameters=[k for k,v in config.items() if k not in exluded_params]
        records+=parameters
    
    # Eliminate all the rendundant parameters and create the csv file
    columns=list(set(records))

    # Save in the results performace folder
    path_csv=os.path.join(data["experiment"]["path_output_rec_performance"],f"CR_cutoff_{data['experiment']['evaluation']['cutoffs']}.csv")
    pd.DataFrame(columns=columns).to_csv(path_csv, index=False)


def add_CR_istance(path_csv,parameters):
    # Load the csv file
    df=pd.read_csv(path_csv)

    # Create a new row with the parameters of the current model
    new_row={k:v for k,v in parameters.items() if k in df.columns}
    
    # Append the new row to the dataframe
    df=df.append(new_row, ignore_index=True)

    # Save the updated dataframe to the csv file
    df.to_csv(path_csv, index=False)


run_experiment(config_path)