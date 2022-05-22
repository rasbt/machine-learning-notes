#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def train_classifier_simple_v2(
    model,
    num_epochs,
    train_loader,
    valid_loader,
    test_loader,
    optimizer,
    device,
    logging_interval=50,
    best_model_save_path=None,
    scheduler=None,
    skip_train_acc=False,
    scheduler_on="valid_acc",
):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    best_valid_acc, best_epoch = -float("inf"), 0

    for epoch in range(num_epochs):

        epoch_start_time = time.time()
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                    f"| Batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"| Loss: {loss:.4f}"
                )

        model.eval()

        elapsed = (time.time() - epoch_start_time) / 60
        print(f"Time / epoch without evaluation: {elapsed:.2f} min")
        with torch.no_grad():  # save memory during inference
            if not skip_train_acc:
                train_acc = compute_accuracy(model, train_loader, device=device).item()
            else:
                train_acc = float("nan")
            valid_acc = compute_accuracy(model, valid_loader, device=device).item()
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc, best_epoch = valid_acc, epoch + 1
                if best_model_save_path:
                    torch.save(model.state_dict(), best_model_save_path)

            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                f"| Train: {train_acc :.2f}% "
                f"| Validation: {valid_acc :.2f}% "
                f"| Best Validation "
                f"(Ep. {best_epoch:03d}): {best_valid_acc :.2f}%"
            )

        elapsed = (time.time() - start_time) / 60
        print(f"Time elapsed: {elapsed:.2f} min")

        if scheduler is not None:

            if scheduler_on == "valid_acc":
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == "minibatch_loss":
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError("Invalid `scheduler_on` choice.")

    elapsed = (time.time() - start_time) / 60
    print(f"Total Training Time: {elapsed:.2f} min")

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f"Test accuracy {test_acc :.2f}%")

    elapsed = (time.time() - start_time) / 60
    print(f"Total Time: {elapsed:.2f} min")

    return minibatch_loss_list, train_acc_list, valid_acc_list


def get_dataloaders_mnist(
    batch_size,
    num_workers=0,
    validation_fraction=None,
    train_transforms=None,
    test_transforms=None,
):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data", train=True, transform=train_transforms, download=True
    )

    valid_dataset = datasets.MNIST(root="data", train=True, transform=test_transforms)

    test_dataset = datasets.MNIST(root="data", train=False, transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 60000)
        train_indices = torch.arange(0, 60000 - num)
        valid_indices = torch.arange(60000 - num, 60000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=valid_sampler,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler,
        )

    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=True,
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader


class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, required=True, help="Which GPU device to use."
    )

    args = parser.parse_args()

    RANDOM_SEED = 123
    BATCH_SIZE = 128
    NUM_EPOCHS = 1
    DEVICE = torch.device(args.device)

    print("torch", torch.__version__)
    print("device", DEVICE)

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5)),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5)),
        ]
    )

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
        batch_size=BATCH_SIZE,
        validation_fraction=0.1,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=2,
    )

    torch.manual_seed(RANDOM_SEED)

    model = LeNet5(num_classes=10, grayscale=True)

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    minibatch_loss_list, train_acc_list, valid_acc_list = train_classifier_simple_v2(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        best_model_save_path=None,
        device=DEVICE,
        scheduler_on="valid_acc",
        logging_interval=100,
    )
