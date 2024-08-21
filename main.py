import os
import torch
import logging
from absl import app, flags
import torch.multiprocessing as mp
from ml_collections.config_flags import config_flags
from utils.file_utils import create_workdir, load_and_override_config, log_and_print
from train import train


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training Configuration.")
flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results.")
flags.DEFINE_integer('distributed', 1, 'Distributed training: 1 to enable, 0 to disable')
flags.DEFINE_integer('log', 1, 'Log in wandb: 1 to enable, 0 to disable')
flags.DEFINE_string('override_param', None, 'Parameter to override the config, e.g., "learning_rate=0.01".')
flags.DEFINE_integer('target_class', None, 'If set, filters the dataset to only include this target class.')
flags.DEFINE_integer('selected_attributes', None, 'If set, it will filter the CelebA dataset to only include this attribute.')

flags.mark_flags_as_required(["workdir", "config", "mode"])


def setup_logging_to_file(workdir):
    log_file_path = open(os.path.join(workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main(argv):
    config = load_and_override_config(FLAGS)
    if FLAGS.mode == "train":
        create_workdir(FLAGS.workdir)
        setup_logging_to_file(FLAGS.workdir)
        world_size = torch.cuda.device_count()
        log_and_print(f"Distributed training\nNumber of gpus: {world_size}")
        mp.spawn(train, args=(config, FLAGS.workdir, FLAGS.log, FLAGS.target_class, FLAGS.selected_attributes), nprocs=world_size)
    else:
        logging.info("No option available")


if __name__ == '__main__':
    app.run(main)