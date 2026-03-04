import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import os
from model import SimpleNet

# Config
with open('config.yaml','r') as f:
    config = yaml.safe_load(f)

# ⭐ STRONG AUGMENTATION (VERY IMPORTANT)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

# Dataset loaders
def get_loaders():

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

    val_split = config['hyperparameters']['val_split']

    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    dataset_train, dataset_val = random_split(
        dataset_train_full,
        [n_train,n_val]
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=64,
        shuffle=False
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=False
    )

    return loader_train, loader_val, loader_test


# Evaluation
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()*labels.size(0)

            _,pred = torch.max(outputs,1)

            total += labels.size(0)
            correct += (pred==labels).sum().item()

    return total_loss/total, 100*correct/total


# Training
def train(model, train_loader, val_loader,
          criterion, optimizer, scheduler, device):

    epochs = 20

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for inputs, labels in tqdm(train_loader):

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
            model,val_loader,criterion,device
        )

        print(f"Epoch {epoch+1}")
        print(f"Train Loss {running_loss/len(train_loader):.3f}")
        print(f"Val Acc {val_acc:.2f}%")

    print("Training finished")


# Main
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader,val_loader,test_loader = get_loaders()

    model = SimpleNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # Cosine-like decay behavior
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=8,
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

    test_loss,test_acc = evaluate(
        model,test_loader,criterion,device
    )

    print("Final Test Accuracy =", test_acc)

    # Save model
    os.makedirs("weights",exist_ok=True)

    torch.save(
        model.state_dict(),
        "weights/checkpoint.pth"
    )

if __name__=="__main__":
    main()