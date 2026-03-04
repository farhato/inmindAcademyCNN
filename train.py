import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import os
from model import SimpleNet


# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Data Augmentation (VERY IMPORTANT FOR ACCURACY)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# Data Loader Function
def get_loaders():

    os.makedirs(config['paths']['train_dir'], exist_ok=True)
    os.makedirs(config['paths']['test_dir'], exist_ok=True)

    dataset_train_full = datasets.CIFAR10(
        root=config['paths']['train_dir'],
        train=True,
        download=True,
        transform=transform
    )

    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'],
        train=False,
        download=True,
        transform=transform
    )

    # Train / Validation split
    val_split = config['hyperparameters']['val_split']

    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    dataset_train, dataset_val = random_split(
        dataset_train_full,
        [n_train, n_val]
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=config['hyperparameters']['num_workers']
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )

    return dataloader_train, dataloader_val, dataloader_test


# Evaluation Function
def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, 100 * correct / total


# Training Function
def train(model, train_loader, val_loader,
          criterion, optimizer, scheduler, device):

    epochs = config['hyperparameters']['epochs']

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        with tqdm(train_loader,
                  desc=f"Epoch {epoch+1}/{epochs}",
                  unit="batch") as progress_bar:

            for inputs, labels in progress_bar:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        scheduler.step()

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss {running_loss/len(train_loader):.3f} | "
            f"Val Loss {val_loss:.3f} | "
            f"Val Acc {val_acc:.2f}%"
        )


# Main Function
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, test_loader = get_loaders()

    model = SimpleNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device
    )

    # Test model
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Accuracy = {test_acc:.2f}%")

    # Save model
    os.makedirs(
        os.path.dirname(config['paths']['model_path']),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        config['paths']['model_path']
    )


if __name__ == "__main__":
    main()