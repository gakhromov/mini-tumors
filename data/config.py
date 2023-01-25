import os

ROOT_PATH = os.getcwd()

# Dataset params
TRAIN_DATASETS = [
    "ColonCancer", 
    # "BrainCancer",
]
INFERENCE_DATASETS = [
    "ColonCancerInference", 
]

IMG_SIZE = (64, 64)

# Bad droplet image detection alg params
HOUGH_CIRCLES = {
    "dp": 1, 
    "minDist": 7, 
    "param1": 150,
    "param2": 20, 
    "minRadius": 10, 
    "pct": 0.15, 
    "resize_size": (50, 50)
}

# Normalisation
PERCENTILE = 95
