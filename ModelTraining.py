import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd

#label_dict = {label.split('_')[0]: idx for idx, label in enumerate(os.listdir("videos"))}
label_dict = {'one': 0, 'one_left': 1,
              'two': 2, 'two_left': 3,
              'three': 4, 'three_left': 5,
              'four': 6, 'four_left': 7,
              'five': 8, 'five_left': 9,
              'rock': 10, 'rock_left':  11}


print(label_dict.keys())
print(len(label_dict.keys()))

labels = pd.read_csv("annotations_file.csv")
#targets = [label_dict[label] for label in labels.iloc[:, 1].values if not "love" in label]
targets = []
for label in labels.iloc[:, 1].values:
    if "love" in label:
        continue
    else:
        targets.append(label_dict[label])

data = np.load("features.npy", allow_pickle=True)
#print(targets)
tensor_x = torch.Tensor(data) # transform to torch tensor
tensor_y = torch.Tensor(targets)

print(tensor_y.shape, tensor_x.shape)

dataset = TensorDataset(tensor_x, tensor_y) # create your datset
#my_dataloader = DataLoader(dataset) # create your dataloade


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset,  batch_size=batch_size)



"""Model stuff comes here"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21*3, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 12)
        )
        self.linear_dropout_stack = nn.Sequential(
            nn.Linear(21*3, 128),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.Linear(32, 12)
        )
        self.cnn_stack = nn.Sequential(
            nn.Conv1d(3, 32, 2, stride=2),
            #nn.MaxPool1d(3),
            nn.Linear(10, 32),
            nn.Dropout(),
            nn.Flatten(), # Why the fuck do i need flatten in sequential module???
            nn.Linear(1024, 12)
        )

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)
        #logits = self.linear_dropout_stack(x)
        logits = self.cnn_stack(torch.permute(x, (0,2,1)))
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()


        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 1024
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), 'model.pt')
