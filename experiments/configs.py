# experiments/configs.py: batch experiment combinations for run.py and benchmark.py

CONFIGS = [
    {"method": "direct", "batch_size": 4, "max_epochs": 50, "backbone": "resnet18"},
    {"method": "direct", "batch_size": 4, "max_epochs": 50, "backbone": "resnet34"},
    {"method": "direct", "batch_size": 4, "max_epochs": 50, "backbone": "resnet50"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 50},
    # {"method": "heatmap", "batch_size": 4, "max_epochs": 50},
    # {"method": "homography", "batch_size": 4, "max_epochs": 50},
]
