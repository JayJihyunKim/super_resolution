import h5py
import torch
import torchvision
import torchvision.transforms as transforms
from config import cifar10_config
from torch.utils.data import Dataset

def load_cifar10_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    batch_size = cifar10_config.BATCH_SIZE
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    image_dataset = {
        "train_loader": trainloader,
        "test_loader": testloader,
        "class_label": classes
    }
    return image_dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['lr'][idx], f['hr'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
