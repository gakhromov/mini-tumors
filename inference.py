import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data import data
import augment_data
import pandas as pd
from tqdm import tqdm

# Argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-mf", "--model-filename", type=str, default="saved_models/best_model.pt",
                    help="filename of the model to be used for inference")
parser.add_argument("-pf", "--predictions-filename", type=str, default="predictions.csv",
                    help="filename under which to save predictions")
args = parser.parse_args()

# Load inference data and model
print("Loading model and data...")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
mean = np.array([0.485, 0.456, 0.406]) # should those be adaptable?
std = np.array([0.229, 0.224, 0.225])
data_transforms = {
    'inference': transforms.Compose([
        transforms.Normalize(mean, std)
    ]),
}
inference_dataset, inference_dataloader = data.load_datasets_inference()
image_datasets = {
    'inference' : inference_dataset,
}
dataloaders = {
    'inference' : inference_dataloader,
}
dataset_sizes = {'inference': len(image_datasets['inference'])}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(args.model_filename)
print("Done!")

# Perform inference
with torch.no_grad():
    predictions = []
    print("Performing predictions.....")
    for batch_idx, images in enumerate(tqdm(dataloaders['inference'])):
        images = data_transforms['inference'](images)
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
    print("Done!")

# Save data
predictions_table = pd.DataFrame(data = {'image' : [f'img{i}.npy' for i in range(dataset_sizes['inference'])], 'prediction' : predictions})
predictions_table.to_csv(args.predictions_filename, index = False)
print(f"Predictions saved to {args.predictions_filename}")