import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 100),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


class LightningModel(L.LightningModule):
    def __init__(self, model=None, learning_rate=None):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


class CustomDataset(Dataset):
    def __init__(self, feature_array, label_array, transform=None):

        self.x = feature_array
        self.y = label_array
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.y.shape[0]


class CustomDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./mnist", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):

        X, y = make_classification(
            n_samples=20000,
            n_features=100,
            n_informative=10,
            n_redundant=40,
            n_repeated=25,
            n_clusters_per_class=5,
            flip_y=0.05,
            class_sep=0.5,
            random_state=123,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=123
        )

        self.train_dataset = CustomDataset(
            feature_array=X_train.astype(np.float32),
            label_array=y_train.astype(np.int64),
        )

        self.val_dataset = CustomDataset(
            feature_array=X_val.astype(np.float32), label_array=y_val.astype(np.int64)
        )

        self.test_dataset = CustomDataset(
            feature_array=X_test.astype(np.float32), label_array=y_test.astype(np.int64)
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        return test_loader
