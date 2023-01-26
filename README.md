# mini-tumors project
Data Science Lab 2022 Project - Mini-Tumors in Nanoliter-Droplets for Personalized Cancer Treatment

## First start
Before you start working with the project, you have to setup the environment for the code. You can do this by following the instructions below:
- If you want to setup the project locally, check the instructions [here](docs/environment.md#instructions-for-the-local-setup).
- If you want to setup the project for the euler cluster, check the instructions [here](docs/environment.md#instructions-for-the-euler-setup).

## Prepare the datasets
- Download and prepare the dataset (instructions [here](docs/preparing_data.md)).
- \[Optional\] If you want to do so, you can augment the training dataset, as this usually helps to produce a more robust model. Check the instructions [here](docs/augmentations.md). However, this code is incomplete and we haven't yet fully integrated the use of augmented data in the training procedure.

## Train the model and use it to perform predictions
- If you want to train the model, follow the instructions for [local run](docs/train.md#instructions-for-local-training) or [run on Euler](docs/train.md#instructions-for-training-on-euler).
- If you want to make the model produce results for an unlabelled dataset (inference), follow the instructions for [local run](docs/inference.md#instructions-for-local-inference) or [run on Euler](docs/inference.md#instructions-for-inference-on-euler).
