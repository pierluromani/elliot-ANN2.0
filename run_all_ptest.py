import builtins

if not hasattr(builtins, "long"):
    builtins.long = int

from elliot.run import run_experiment

config_path = "config_stat_test/target_t_0_05.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_25.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_50.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_75.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_95.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_05_minh.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_25_minh.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_50_minh.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_75_minh.yaml"
run_experiment(config_path)

config_path = "config_stat_test/target_t_0_95_minh.yaml"
run_experiment(config_path)
