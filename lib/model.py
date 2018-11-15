from __future__ import print_function

import os
import time
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from .dataset import ProteinDataset

__resnet_dict__ = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}

def load_resnet(num_layers, pretrained):
    assert num_layers in __resnet_dict__.keys()
    model = __resnet_dict__[num_layers](**kwargs)
    for param in model.parameters():
        param.requires_grad = False
    return model

def update_phase(phase, model, scheduler):
    if phase == 'train':
        scheduler.step()
        model.train() # set model to training mode
    else:
        model.eval() # set model to evaluate mode

def train_single_iter(phase, model, criterion, optimizer, inputs, labels):
    optimizer.zero_grad()

    # forward pass: track history if only in train
    with torch.set_grad_enabled(phase == 'train'):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
            loss.backward()
            optimizer.step()

    # statistics
    running_loss = loss.item() * inputs.size(0)
    return running_loss

def train_single_epoch(model, criterion, optimizer, scheduler, loaders, sizes, device):
    epoch_loss = {'train': 0.0, 'eval': 0.0}
    for phase in ['train', 'eval']:
        update_phase(phase, model, scheduler)

        running_loss = 0.0
        
        # Iterate over data.
        for inputs, labels in loaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            running_loss += train_single_iter(phase, model, criterion, optimizer, inputs, labels)

        epoch_loss[phase] = running_loss / sizes[phase]

        print('{} Loss: {:.4f}'.format(phase, epoch_loss[phase]))

    return epoch_loss

def train_model(model, criterion, optimizer, scheduler, num_epochs, loaders, sizes, device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train single epoch
        epoch_loss = train_single_epoch(model, criterion, optimizer, scheduler, loaders, sizes, device)
        if epoch_loss['eval'] < best_loss:
            best_loss = epoch_loss['eval']
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class MultiLabelModel(nn.Module):
    def __init__(self, base, base_out_features, num_classes):
        self.base = base
        self.num_classes = num_classes
        self.final_fc = nn.Linear(base_out_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        x = self.final_fc(x)
        x = self.sigmoid(x)
        return x


if __name__ = 'main':
    base_num_layers = 18
    num_classes = 28
    lr = 0.001
    momentum = 0.9
    num_epochs = 2

    root_dir = '/home/adil/dev/kaggle-protein-classification/data'
    csv_file = os.path.join(root_dir, 'train.csv')

    base = load_resnet(base_num_layers, True)
    model = MultiLabelModel(base, base.fc.out_features, num_classes)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

    # load data