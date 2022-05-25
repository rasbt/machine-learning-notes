import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_transform=None,
        test_transform=None,
        num_workers=4,
        data_path="./",
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.custom_train_transform = train_transform
        self.custom_test_transform = test_transform

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, download=True)
        return

    def setup(self, stage=None):

        if self.custom_train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((70, 70)),
                    transforms.RandomCrop((64, 64)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.train_transform = self.custom_train_transform

        if self.custom_train_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize((70, 70)),
                    transforms.CenterCrop((64, 64)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.test_transform = self.custom_test_transform

        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )

        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )

        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
