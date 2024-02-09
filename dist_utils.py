import os
import torch
from torch.distributed import init_process_group


def ddp_setup(
        rank,
        world_size,
        master_addr="localhost",
        master_port="12365"
    ):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)