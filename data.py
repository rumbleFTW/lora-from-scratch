from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms


from config import batch_size

train_dataset = datasets.MNIST(
    "dataset/", transform=transforms.ToTensor(), download=True, train=True
)

test_dataset = datasets.MNIST(
    "dataset/", transform=transforms.ToTensor(), download=True, train=False
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
