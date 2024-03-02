import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



# This is a custom transform to normalize the features
class CustomNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, numpy_array):
        normalized_data = (numpy_array - self.mean) / self.std

        tensor = torch.from_numpy(normalized_data)
        return tensor
    


# This is a custom transform to convert the the string label to integers
class LabelTransform:
    def __init__(self, label_set):
        # Map each of the first 10 letters of the alphabet to an index
        self.label_to_index = {label: index for index, label in enumerate(label_set)}

    def __call__(self, label):
        return self.label_to_index[label]
    


def train(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader) 

    train_loss = 0

    # Sets the model in 'training' mode
    model.train()                                        #  This helps inform layers such as Dropout and BatchNorm, which are
                                                         #   designed to behave differently during training and evaluation.
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)               # '.to(device)' allows you to move the data to the device you are using (ex. GPU memory)

        # Compute prediction error
        pred = model(X)
        # print(X.shape, y.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()                            # This command resets the gradients (pytorch acumulates them by default)

        train_loss += loss.item()

    print(f"loss: {loss:>7f}")
    return train_loss / num_batches         # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def test_loss(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)                   

    # Sets the model in 'evaluation' mode
    model.eval()

    test_loss = 0

    # Temporarily disable gradients computation
    with torch.no_grad():                                 # This disables the memorization of the activation functions during the forward pass
        for X, y in dataloader:                           #  (they are used in the backprop phase to compute the gradients)
            X, y = X.to(device), y.to(device)     
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss



def test_accuracy(dataloader, model, device):
    size = len(dataloader.dataset)


    # Sets the model in 'evaluation' mode
    model.eval()

    correct = 0

    # Temporarily disable gradients computation
    with torch.no_grad():                                 # This disables the memorization of the activation functions during the forward pass
        for X, y in dataloader:                           #  (they are used in the backprop phase to compute the gradients)
            X, y = X.to(device), y.to(device)     
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()     # The prediction is the argmax of the last layer
    correct /= size 
    return correct

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%")
    # return correct

