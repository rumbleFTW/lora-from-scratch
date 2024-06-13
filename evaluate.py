import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import CNN
import config
from data import test_dataloader


def main():
    device = torch.device(config.device)
    network = CNN()
    network.load_state_dict(torch.load("checkpoints/cnn.pth"))
    network = network.to(device)
    # Eval:

    correct_predictions = 0
    total_predictions = 0
    progress = tqdm(test_dataloader)
    with torch.no_grad():
        for features, labels in progress:
            features, labels = features.to(device), labels.to(device)
            predicted = network(features)
            preds = torch.tensor(
                [torch.argmax(elem).item() for elem in F.softmax(predicted, dim=1)]
            )
            correct_predictions += (preds.cpu() == labels.cpu()).sum().item()
            total_predictions += labels.size(0)
            accuracy = correct_predictions / total_predictions
            progress.set_description(f"Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
