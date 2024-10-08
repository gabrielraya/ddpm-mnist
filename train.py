import os
import math
import time
import torch
import logging
from datasets import load_data, rescaling_inv
from utils.file_utils import create_workdir, log_and_print, setup_wandb
from utils.dist_utils import ddp_setup
from plots import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from models.ddpm import DDPM
from diffusion.diffusion_lb import GaussianDiffusion
from Scheduler import GradualWarmupScheduler
from diffusion.losses import get_ddpm_loss_fn
from diffusion import sampling
import wandb
from torchvision.utils import make_grid


def initialize_directories(workdir):
    """Initializes the sample and model directories."""
    sample_dir = os.path.join(workdir, "samples")
    model_dir = os.path.join(workdir, "model")
    create_workdir(sample_dir)
    create_workdir(model_dir)
    return sample_dir, model_dir


def count_parameters(model):
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(rank, config, workdir, log=True, target_class=None, selected_attributes=None):
    if log and rank == 0:
        setup_wandb(config, rank)

    world_size = torch.cuda.device_count()
    ddp_setup(rank, world_size)
    log_and_print(f"Initializing process group: rank={rank}, world_size={world_size}")

    # set device
    device = torch.device("cuda", rank)

    # experimental settings
    dataset_name = config.data.dataset.lower()
    sample_dir, model_dir = initialize_directories(workdir)

    # datasets
    train_loader, test_loader, sampler = load_data(config,
                                                   data_path="../datasets/",
                                                   num_workers=config.training.numworkers,
                                                   evaluation=False,
                                                   distributed=config.training.distributed,
                                                   target_class=target_class,
                                                   selected_attributes=selected_attributes
                                                   )

    # Save a batch of training samples for visualization
    x, y = next(iter(train_loader))
    n = config.sampling.generated_batch if x.shape[0] >= config.sampling.generated_batch else x.shape[0]
    save_image(rescaling_inv(x[:n]), workdir=workdir, pos="square", name="{}_data_samples".format(dataset_name))

    log_and_print("Training U-Net model")
    model = DDPM(config)
    param_count = count_parameters(model)
    model.to(device)

    # Resume training when intermediate checkpoints are detected
    last_epoch = 0
    if os.path.exists(os.path.join(model_dir, 'last_epoch.pt')):
        last_epoch = torch.load(os.path.join(model_dir, 'last_epoch.pt'))['last_epoch']
        # load checkpoints
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(os.path.join(model_dir, f'ckpt_{last_epoch}_checkpoint.pt'), map_location=map_location)
        model.load_state_dict(checkpoint['net'])

    # setup the diffusion process
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)  # defines the diffusion process

    # Distributed Data parallel settings
    model = DDP(model, device_ids=[rank], output_device=rank)

    # optimizer settings
    optimizer = torch.optim.RAdam(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)

    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.training.n_epochs,
        eta_min=0,
        last_epoch=-1
    )

    log_and_print("Starting training loop at epoch %d." % (last_epoch,))

    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=config.optim.multiplier,
        warm_epoch=config.training.n_epochs // 10,
        after_scheduler=cosineScheduler,
        last_epoch=last_epoch
    )

    if last_epoch != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # training
    logging.info("Starting training loop at epoch %d." % (last_epoch,))
    cnt = torch.cuda.device_count()

    loss_fn = get_ddpm_loss_fn(diffusion, train=True)
    loss_fn_val = get_ddpm_loss_fn(diffusion, train=False)

    for epoch in range(last_epoch, config.training.n_epochs):
        start_time = time.time()

        sampler.set_epoch(epoch)
        avg_loss = 0
        n_iter = 0

        for batch_images, _ in train_loader:
            optimizer.zero_grad()
            batch = batch_images.to(device)
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            n_iter += 1
        avg_loss = avg_loss / n_iter
        end_time = time.time()
        epoch_time = end_time - start_time

        log_and_print("epoch: %d, avg_training_loss: %.5e, batch per device:  %d, LR=%.7e, epoch_time=%.2f seconds, "
                      "Parameter Count: %d" % (epoch, avg_loss, batch.shape[0],
                                               optimizer.state_dict()['param_groups'][0]["lr"],
                                               epoch_time, param_count))

        if log and rank == 0:
            # 🐝 Log train metrics to wandb
            metrics = {"avg train loss": avg_loss}
            wandb.log(metrics, step=epoch)

        warmUpScheduler.step()

        # evaluation and save checkpoint
        if (epoch + 1) % config.training.eval_freq == 0:
            x, labels = next(iter(test_loader))
            with torch.no_grad():
                x = x.to(device)
                eval_loss = loss_fn_val(model, x)
                log_and_print("epoch: %d, eval_loss: %.5e, batch per device:  %d" %
                              (epoch, eval_loss.item(), batch_images.shape[0],))
                if log and rank == 0:
                    # 🐝 Log train and validation metrics to wandb
                    val_metrics = {"eval loss": eval_loss.item()}
                    wandb.log(val_metrics, step=epoch)

        if (epoch + 1) % config.training.snapshot_freq == 0:
            # generated unconditional samples
            all_samples = []
            each_device_batch = config.sampling.generated_batch // cnt
            # TODO: adapt this to multigpu generation
            if rank == 0:
                with torch.no_grad():
                    # force embbedding to be a zero vector for unconditional generation
                    C, H, W = config.data.num_channels, config.data.image_size, config.data.image_size
                    sampling_shape = (config.sampling.generated_batch, C, H, W)
                    uncond_samples = sampling.sampling_fn(config, diffusion, model, sampling_shape, rescaling_inv)
                    uncond_samples = torch.clip(uncond_samples * 255, 0, 255).int()
                    name = "{}_{}_uncond_generated_{}.png".format(config.model.name, dataset_name, epoch + 1)
                    save_image(uncond_samples, sample_dir, n=config.sampling.generated_batch, pos="square", name=name)

                if log:
                    # 🐝 2️⃣ Log metrics from your script to W&B
                    n = int(math.sqrt(config.sampling.generated_batch))
                    sample_grid = make_grid(uncond_samples.clone().detach()[:n ** 2], nrow=n, padding=1)
                    wandb.log({"generated_image": [wandb.Image(sample_grid.float() / 255.0, caption=f"Epoch {epoch}")]})

            # save checkpoints
            checkpoint = {
                'net': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': warmUpScheduler.state_dict()
            }
            torch.save({'last_epoch': epoch + 1}, os.path.join(model_dir, 'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(model_dir, f'ckpt_{epoch + 1}_checkpoint.pt'))

        torch.cuda.empty_cache()

    if log and rank == 0:
        # 🐝 Close your wandb run
        wandb.finish()

    # end distributed training
    destroy_process_group()
