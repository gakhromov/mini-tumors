# Instructions for local inference
[<< go back to the main README](../README.md)
1. Make sure that the inference dataset is properly set up. All the necessary data resulting from the prepare_dataset.py script should be in the 'data/inference' subfolder.
2. Run the inference script with `pipenv run python inference.py`. If desired, you can also manually specify the file name of the model to be used (by default, the best model from our training process will be selected) and the file name under which to save predictions (by default, saved as 'predictions.csv'): for example, you can run `pipenv run python inference.py --model-filename "saved_models/model2.pt" --predictions-filename "predictions2.csv"`.
# Instructions for inference on Euler
[<< go back to the main README](../README.md)
1. TO BE COMPLETED