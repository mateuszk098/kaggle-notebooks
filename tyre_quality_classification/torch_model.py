import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()


class MaxDepthPool2d(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        shape = x.shape
        channels = shape[1] // self.pool_size
        new_shape = (shape[0], channels, self.pool_size, *shape[-2:])
        return torch.amax(x.view(new_shape), dim=2)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=16):
        super().__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.feed_forward = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_channels, squeeze_channels),
            nn.ReLU(),
            nn.Linear(squeeze_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        calibration = self.feed_forward(x)
        return x * calibration.view(-1, x.shape[1], 1, 1)


class ResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, squeeze_active=False):
        super().__init__()
        pad = kernel_size // 2
        self.squeeze_active = squeeze_active
        self.squeeze_excitation = SqueezeExcitation(out_channels)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut_connection = nn.Sequential()
        if not in_channels == out_channels or stride > 1:
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x_residual = self.feed_forward(x)
        x_shortcut = self.shortcut_connection(x)
        residual_output = F.mish(x_residual + x_shortcut)

        if self.squeeze_active:
            return self.squeeze_excitation(residual_output) + x_shortcut

        return residual_output


class ImagePathsDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        if not len(paths) == len(labels):
            raise ValueError(f"'paths' length must be equal to 'labels' length")
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        label = self.labels[index]
        if self.transform is not None:
            return self.transform(img), label
        return img, label


def get_train_valid_subsets(data_path, /, valid_ratio=0.2, seed=None):
    defective_paths = glob.glob(str(data_path / "defective/*"))
    good_paths = glob.glob(str(data_path / "good/*"))
    paths = np.concatenate((defective_paths, good_paths))

    defective_labels = [0] * len(defective_paths)
    good_labels = [1] * len(good_paths)
    labels = np.concatenate((defective_labels, good_labels))

    return train_test_split(
        paths, labels, test_size=valid_ratio, random_state=seed, stratify=labels
    )


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    # print("Before loop")
    for batch, (X, y) in enumerate(dataloader):
        # y = y.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)
        # print(y.max())
        # print("After x, y to device")
        # Compute prediction error.
        pred = model(X).squeeze()
        #  print("After pred")
        loss = loss_fn(pred, y)
        # print("After loss")
        # Backpropagation
        loss.backward()
        #  print("After backprop")
        optimizer.step()
        #   print("After optimizer")
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    dataset_path = Path("data/tyres")
    train_paths, valid_paths, train_labels, valid_labels = get_train_valid_subsets(
        dataset_path, valid_ratio=0.125, seed=42
    )

    cnn = nn.Sequential(
        #
        nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.Mish(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(64),
        #
        ResidualConnection(64, 128, kernel_size=3, stride=2, squeeze_active=True),
        MaxDepthPool2d(pool_size=2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=64),
        #
        nn.Flatten(),
        #
        nn.Linear(64 * 14 * 14, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.Mish(),
        nn.Dropout1d(0.4),
        #
        nn.Linear(256, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.Mish(),
        nn.Dropout1d(0.4),
        #
        nn.Linear(256, 1),
    ).to(device)

    transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = ImagePathsDataset(train_paths, train_labels, transform=transform)
    valid_dataset = ImagePathsDataset(valid_paths, valid_labels, transform=transform)

    batch_size = 32
    num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # type: ignore
        persistent_workers=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # type: ignore
        persistent_workers=True,
        pin_memory=True,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, cnn, loss_fn, optimizer)
    print("Done!")


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    main()
    print(time.perf_counter() - t0)
