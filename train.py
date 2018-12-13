import os
import pandas as pd
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import scipy.io

data_dir = '/Users/tttk1/Desktop/deepsat-sat6/'


class SatelliteDataset(Dataset):
    def __init__(self, X_csv, y_csv, transform=None):
        self.instances = pd.read_csv(X_csv, nrows=5000)
        self.labels = pd.read_csv(y_csv, nrows=5000)
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = self.instances.iloc[idx].values.reshape(-1, 4, 28, 28).clip(0, 255).astype(np.uint8).squeeze(axis=0)
        label = self._get_labels(self.labels.iloc[idx].values)
        return {'chip': img, 'label': label}

    def _get_labels(self, row_values):
        annotations = {'building': 0, 'barren_land': 1,'trees': 2, 'grassland': 3, 'road': 4, 'water': 5}
        labels = [list(annotations.values())[i] for i, x in enumerate(row_values) if x == 1]
        return labels[0]

training_set = SatelliteDataset(X_csv=data_dir+'X_train_sat6.csv', y_csv=data_dir+'y_train_sat6.csv')
validation_set = SatelliteDataset(X_csv=data_dir+'X_test_sat6.csv', y_csv=data_dir+'y_test_sat6.csv')

tfms = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomVerticalFlip(),
    tv.transforms.RandomRotation(90),
    tv.transforms.ToTensor()
])

train_loader = DataLoader(
    training_set,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    validation_set,
    batch_size=64,
    shuffle=True,
    num_workers=4
)


class SmallCNN(t.nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=3
            ),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(),
            t.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3
            ),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(),
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(),
            t.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3
            ),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(),
        )
        self.pool = t.nn.MaxPool2d(2)
        self.fc1 = t.nn.Linear(1024, 512)
        self.bn = t.nn.BatchNorm1d(512)
        self.drop_out = t.nn.Dropout2d(0.2)
        self.fc2 = t.nn.Linear(512, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.bn(x)
        x = t.nn.functional.relu(x)
        x = self.drop_out(x)
        output = self.fc2(x)
        return output

model = SmallCNN()
optim = t.optim.SGD(lr=0.001, momentum=0.9, params=model.parameters())
crit = t.nn.CrossEntropyLoss()

EPOCHS = 20
for e in range(EPOCHS):
    for s, inst in enumerate(train_loader):
        chip = inst['chip'].float()
        output = model(chip)
        loss = crit(output, inst['label'])
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'Epoch {e+1} Step {s+1} Loss {loss.item()}')

model.eval()
with t.no_grad():
    total = 0
    correct = 0
    for s, inst in enumerate(val_loader):
        output = model(inst['chip'].float())
        _, predicted = t.max(output.data, 1)
        total += inst['label'].size(0)
        correct += (predicted == inst['label']).sum().item()
    print(f'Accuracy {(100 * correct / total)}')