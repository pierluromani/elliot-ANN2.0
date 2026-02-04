import pandas as pd
import os
import yaml

def create_CR_dataset(config_path: str, excluded_params: list):
    # Parse the configuration file 
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)

    # Create a list of all the parameters to be recorded
    records = ["CR", "Model"]
    
    # Iterate all the models and collect their parameters
    for model_name, config in data["experiment"]["models"].items():
        parameters=[k for k,v in config.items() if k not in excluded_params]
        records+=parameters
    
    # Eliminate redundant parameters
    columns = list(set(records))

    # Save in the results performance folder
    if "csv_path" in data["experiment"]:
         path_csv = data["experiment"]["csv_path"]
    else:
        path_csv = os.path.join(data["experiment"]["path_output_rec_performance"], 
                                f"CR_cutoff_{data['experiment']['evaluation']['cutoffs']}.csv")
                                
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    
    if not os.path.exists(path_csv):
        pd.DataFrame(columns=columns).to_csv(path_csv, index=False)


def add_CR_instance(path_csv, parameters: dict):
    if not path_csv:
        return

    # Load the csv file
    try:
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        # If file not found, creating a new dataframe with parameters keys as columns
        df = pd.DataFrame(columns=list(parameters.keys()))

    # Create a new DataFrame with the parameters of the current model
    # Ensure we only include keys that are actual columns in the CSV
    # If the CSV was just created or keys are missing, we might want to be flexible, 
    # but based on previous code, we filter. 
    # To be safe against new columns:
    
    # If we want to allow new columns to be added dynamically:
    # new_row_df = pd.DataFrame([parameters])
    # df = pd.concat([df, new_row_df], ignore_index=True)
    
    # Sticking to the strict schema logic from the user's previous code:
    if not df.empty:
        new_data = {k: v for k, v in parameters.items() if k in df.columns}
    else:
         new_data = parameters

    new_row_df = pd.DataFrame([new_data])
    
    # Concatenate handles the missing columns by filling them with NaN
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Save the updated dataframe to the csv file
    df.to_csv(path_csv, index=False)
