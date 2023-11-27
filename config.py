"""
This file contains the configuration for the training process.
"""
from pathlib import Path

def get_config():
    """
    Returns the configuration dictionary for the training process. 

    :return: the configuration dictionary
    """
    return {
        # Batch size for training
        "batch_size": 8,
        # Number of epochs to train for
        "num_epochs": 40,
        # Learning rate
        "lr": 10**-4,
        # Sequence length, should be more than the longest sentence in the dataset (printed at the beginning)
        "seq_len": 350,
        # Dimension of the model, 512 is the default mentioned in the paper
        "d_model": 512,
        "datasource": 'opus_books',
        # Source language of the dataset
        "lang_src": "en",
        # Target language of the dataset
        "lang_tgt": "it",
        "model_basename": "tmodel_",
        "model_folder": "weights",
        # Whether to use the latest weights file in the weights folder
        # set to None to not use any weights and start training from scratch
        # set to 'latest' to use the latest weights file
        "preload": "latest",  # None or 'latest'
        # Tokenizer file, this is where the tokenizer will be saved
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Returns the path to the weights file for the given epoch.

    :param config: the configuration dictionary
    :param epoch: the epoch number

    :return: the path to the weights file
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    """
    Finds the latest weights file in the weights folder, when preload is set to latest
    this function will be used to find the latest weights file.

    :param config: the configuration dictionary
    :return: the path to the latest weights file
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))

    if len(weights_files) == 0:
        return None

    weights_files.sort()

    return str(weights_files[-1])
