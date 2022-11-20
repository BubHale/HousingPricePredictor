"""
Author: Samuel Hale
Creation Date: 11/19/22
Purpose: Looks at housing data and predicts the price depending on coordinates, age, distance from certain commodities
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

"""
IMPORT DATA
CLEAN IT
PREPARE IT FOR TRAINING 
"""

df_raw = pd.read_csv(r'C:\Users\wizar\PycharmProjects\HousingRegression\Real estate valuation data set.csv')

input_cols = ['No', 'X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores',
              'X5 latitude', 'X6 longitude']
output_cols = ['Y house price of unit area']
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

#SEND IN RAW DATAFRAME AND GET NUMPY ARRAYS BACK
inputs_array, targets_array = dataframe_to_arrays(df_raw)

inputs_array = inputs_array.astype(np.float32)
targets_array = targets_array.astype(np.float32)

def NormalizeInput(input, cols):
    i = 0
    while i < len(cols):
        input[:, i] = input[:, i] / input[:, i].max()
        i = i + 1
    return input


def DeNormalizeInput(input, cols):
    i = 0
    while i < len(cols):
        input[:, i] = input[:, i] * input[:, i].max()
        i = i + 1
    return input
inputs_array = NormalizeInput(inputs_array, input_cols)
targets_array = NormalizeInput(targets_array, output_cols)

inputs = torch.from_numpy(inputs_array) #414x7
targets = torch.from_numpy(targets_array) #414x1

dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [344, 70])
batch_size = 16

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2)

#MODEL DEFINITION

class HousingPricePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        inputs, targets = batch
        out = self(inputs)  # Generate predictions
        loss = F.mse_loss(out, targets)  # Calculate loss
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)  # Generate predictions
        loss = F.mse_loss(out, targets)  # Calculate loss
        acc = accuracy(out, targets)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


model = HousingPricePredictor(7, 1)

"""# TRAINING"""


def accuracy(outputs, targets):
    acc = 1 - (abs(torch.sum(outputs) - torch.sum(targets)) / torch.sum(outputs)) #Take the opposite of percent error to be accuracy
    return acc


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


opt_func = torch.optim.SGD

result = evaluate(model, val_dl)

print(result)

history = fit(2000, 1e-5, model, train_dl, val_dl, opt_func)

torch.save(model.state_dict(), 'HousingRegressionParameters')

"""
MAKE PREDICTIONS ON SINGLE SAMPLE
"""

def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)    
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)

input, target = val_ds[0]
predict_single(input, target, model)

input, target = val_ds[10]
predict_single(input, target, model)

input, target = val_ds[20]
predict_single(input, target, model)