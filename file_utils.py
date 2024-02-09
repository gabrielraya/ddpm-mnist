import re
import os
import torch
import wandb
import logging


def create_workdir(workdir):
    try:
        os.makedirs(workdir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating directory {workdir}: {e}")
        raise


def set_nested_config_value(config_dict, key_path, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        if key not in config_dict:
            raise KeyError(f"Key {key} not found in configuration.")
        config_dict = config_dict[key]

    final_key = keys[-1]
    if final_key not in config_dict:
        raise KeyError(f"Final key {final_key} not found in configuration.")

    # Convert the string value to a float or int if appropriate
    if value.isdigit():  # Integer
        value = int(value)
    elif re.match(r"^-?\d+\.\d+$", value):  # Float
        value = float(value)
    elif value.lower() in ['true', 'false']:  # Boolean
        value = value.lower() == 'true'

    config_dict[final_key] = value


def override_config(config, override_string):
    """
    Override configuration parameters based on a comma-separated list of key-value pairs.
    Example override_string: "training.batch_size=64,optim.lr=0.001"
    """
    overrides = override_string.split(',')
    for override in overrides:
        key, value = override.split('=')
        set_nested_config_value(config, key, value)


def load_and_override_config(FLAGS):
    """
    Overrides config file values for quick testing
    """
    config = FLAGS.config
    if FLAGS.override_param:
        override_config(config, FLAGS.override_param)
    return config


# =========================
# Logging and Printing
# =========================
def log_and_print(*args):
    """Helper function to log both in interactive and non-interactive mode.

    Accepts variable number of arguments, similar to print() or logging.info().
    It will concatenate the arguments if more than one is provided.
    """
    if len(args) == 1:
        message = args[0]
    else:
        message = ' '.join(str(arg) for arg in args)

    logging.info(message)
    print(message)


def setup_wandb(config, rank, mode="train"):
    """"Initialize the wandb project based on the given mode and config file"""
    project_name = "et-diff" if mode == "train" else "et-diff-eval"

    wandb.init(
        project=project_name,
        name="{}_{}_{}".format(config.model.name, config.data.dataset.lower(), config.training.n_epochs),
        config=dict(config)
    )
    # Set the desired GPU for WandB to avoid tracking all the gpus
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)