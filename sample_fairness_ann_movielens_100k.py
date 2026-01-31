from elliot.run import run_experiment

# start from movielens 100k dataset
print("Done! We are now starting experiment on movielens 1M dataset")
run_experiment("config_run_experiments/item_experiment_movielens_100k.yml")