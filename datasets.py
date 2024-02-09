import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Callable


def rescaling(x):
    """Rescale data to [-1, 1]. Assumes data in [0,1]."""
    return x * 2. - 1.


def rescaling_inv(x):
    """Rescale data back from [-1, 1] to [0,1]."""
    return .5 * x + .5


def identity(x):
    """Identity transformation."""
    return x


def get_transform(dataset: str = "MNIST", image_size=32, flips=False):
    """Get transformations for the specified dataset."""
    if flips:
        flip_transform = transforms.RandomHorizontalFlip()
    else:
        flip_transform = transforms.Lambda(identity)

    if dataset == 'CelebA':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            rescaling,
            flip_transform
        ])
        trans_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            rescaling
        ])
    else: # Default to MNIST transformation
        trans = transforms.Compose([transforms.ToTensor(), flip_transform, rescaling])
        trans_eval = transforms.Compose([transforms.ToTensor(), rescaling])

    return trans, trans_eval


def load_data(config=None, data_path="../datasets", num_workers=4, evaluation=False, distributed=True):
    """Load dataset based on the configuration."""
    # Get transformation functions
    trans, trans_eval = get_transform(config.data.dataset, config.data.image_size, config.data.random_flip)

    # Get dataset class
    dataset_cls = getattr(datasets, config.data.dataset)

    # Load dataset with given transformations
    if config.data.dataset == "CelebA":
        train_dataset = dataset_cls(root=data_path, transform=trans, split='train', download=True)
        test_dataset = dataset_cls(root=data_path, transform=trans_eval, split='valid', download=True)
    else:
        train_dataset = dataset_cls(root=data_path, transform=trans, train=True, download=True)
        test_dataset = dataset_cls(root=data_path, transform=trans_eval, train=False, download=True)

    # Setup samplers
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    test_sampler = torch.utils.data.SequentialSampler(test_dataset) if distributed else None

    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

    # Configure data loaders
    if distributed:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

    return train_loader, test_loader, train_sampler
