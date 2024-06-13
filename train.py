import torch
import torch.nn as nn

from tqdm import tqdm
import wandb

from model import CNN
from data import train_dataloader
from config import *


def main():

    wandb.init(
        project="cnn-mnist",
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "MNIST",
            "epochs": num_epochs,
        },
    )

    # Training loop
    network = CNN().to(device=device)
    params = sum(p.numel() for p in network.parameters())
    print(f"Number of trainable parameters: {params / 1_000_000:.1f}M")
    optim = torch.optim.Adam(network.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    progress = tqdm(range(num_epochs))
    for epoch in progress:
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.to(device)
            predicted = network(features)
            loss = criterion(predicted, labels)

            optim.zero_grad()
            loss.backward()

            optim.step()
        progress.set_description(f"loss: {loss.item()}")
        wandb.log({"loss": loss})

    wandb.finish()
    torch.save(network.state_dict(), "checkpoints/cnn.pth")


if __name__ == "__main__":
    main()
