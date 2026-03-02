import torch
from torchvision import datasets, transforms
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _get_mnist_dataloaders(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    # Return features and labels directly to manipulate them for split/permuted
    x = dataset.data.float() / 255.0
    x = (x - 0.1307) / 0.3081
    x = x.view(-1, 28 * 28)
    y = dataset.targets
    return x, y

def _get_cifar10_dataloaders(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
    ])
    dataset = datasets.CIFAR10('data', train=train, download=True, transform=transform)
    x = torch.tensor(dataset.data).float().permute(0, 3, 1, 2) / 255.0  # NCHW
    
    # Apply normalization manually to the whole tensor
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2615]).view(1, 3, 1, 1)
    x = (x - mean) / std

    y = torch.tensor(dataset.targets)
    return x, y

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices.
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            self.n_batches += 1
            
    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        idx = self.dataset_len if self.i + self.batch_size > self.dataset_len else self.i + self.batch_size
        
        if self.indices is not None:
            indices = self.indices[self.i:idx]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(t[self.i:idx] for t in self.tensors)
            
        self.i = idx
        return batch
        
    def __len__(self):
        return self.n_batches


def get_split_mnist(batch_size=10):
    """
    Returns 5 tasks of Split-MNIST.
    Tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
    """
    train_x, train_y = _get_mnist_dataloaders(batch_size, train=True)
    test_x, test_y = _get_mnist_dataloaders(batch_size, train=False)

    tasks_train = []
    tasks_test = []

    for i in range(5):
        c1, c2 = 2 * i, 2 * i + 1
        
        # Train
        train_mask = (train_y == c1) | (train_y == c2)
        idx = train_mask.nonzero().squeeze()
        tx, ty = train_x[idx], train_y[idx]
        ty[ty == c1] = 0
        ty[ty == c2] = 1
        tasks_train.append(FastTensorDataLoader(tx, ty, batch_size=batch_size, shuffle=True))
        
        # Test
        test_mask = (test_y == c1) | (test_y == c2)
        idx = test_mask.nonzero().squeeze()
        tx_t, ty_t = test_x[idx], test_y[idx]
        ty_t[ty_t == c1] = 0
        ty_t[ty_t == c2] = 1
        tasks_test.append(FastTensorDataLoader(tx_t, ty_t, batch_size=batch_size, shuffle=False))

    return tasks_train, tasks_test, 2 # number of classes per task

def get_split_cifar10(batch_size=10):
    """
    Returns 5 tasks of Split-CIFAR-10.
    """
    train_x, train_y = _get_cifar10_dataloaders(batch_size, train=True)
    test_x, test_y = _get_cifar10_dataloaders(batch_size, train=False)

    tasks_train = []
    tasks_test = []

    for i in range(5):
        c1, c2 = 2 * i, 2 * i + 1
        
        # Train
        train_mask = (train_y == c1) | (train_y == c2)
        idx = train_mask.nonzero().squeeze()
        tx, ty = train_x[idx], train_y[idx]
        ty[ty == c1] = 0
        ty[ty == c2] = 1
        tasks_train.append(FastTensorDataLoader(tx, ty, batch_size=batch_size, shuffle=True))
        
        # Test
        test_mask = (test_y == c1) | (test_y == c2)
        idx = test_mask.nonzero().squeeze()
        tx_t, ty_t = test_x[idx], test_y[idx]
        ty_t[ty_t == c1] = 0
        ty_t[ty_t == c2] = 1
        tasks_test.append(FastTensorDataLoader(tx_t, ty_t, batch_size=batch_size, shuffle=False))

    return tasks_train, tasks_test, 2


def get_permuted_mnist(batch_size=10, n_tasks=20, seed=0):
    """
    Returns `n_tasks` tasks of Permuted-MNIST.
    Uses a fixed random generator just for generating permutations.
    """
    train_x, train_y = _get_mnist_dataloaders(batch_size, train=True)
    test_x, test_y = _get_mnist_dataloaders(batch_size, train=False)

    tasks_train = []
    tasks_test = []

    # Local RNG for permutations
    rng = np.random.RandomState(seed)

    for i in range(n_tasks):
        if i == 0:
            perm = np.arange(28 * 28)
        else:
            perm = rng.permutation(28 * 28)
            
        ptx = train_x[:, perm]
        pty = train_y.clone()
        tasks_train.append(FastTensorDataLoader(ptx, pty, batch_size=batch_size, shuffle=True))
        
        pttx = test_x[:, perm]
        ptty = test_y.clone()
        tasks_test.append(FastTensorDataLoader(pttx, ptty, batch_size=batch_size, shuffle=False))

    return tasks_train, tasks_test, 10
