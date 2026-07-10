# experiments/configs.py: batch experiment combinations for run.py and benchmark.py

CONFIGS = [
    {"method": "homography", "batch_size": 4, "max_epochs": 50},
    {"method": "direct",     "batch_size": 4, "max_epochs": 50},
]
