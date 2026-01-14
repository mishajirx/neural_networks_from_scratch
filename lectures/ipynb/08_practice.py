# train a model on olivetti faces with pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_olivetti_faces, load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt


# read in resnet models
# read in vision transformer models
# test mlp
# test a few datasets
# talk about segmentation models


# read the dataset
olivetti_faces = fetch_olivetti_faces()



# prepare the data
X = olivetti_faces.images

# normalize data
X = (X - X.mean()) / X.std()


y = olivetti_faces.target

# convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # add channel dimension
X_tensor.shape
y_tensor = torch.tensor(y, dtype=torch.long)

# make training and test splits
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# create a dataset and dataloader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# next(iter(train_dataloader))[0].shape # B x C x H x W = 16 x 1 x 64 x 64
# next(iter(train_dataloader))[1].shape # B = 16

# define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.feature_extractor = nn.Sequential( # B x 1 x 64 x 64
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # output: B x 8 x 64 x 64
            nn.ReLU(), # B x 8 x 64 x 64
            nn.AvgPool2d(kernel_size=2, stride=2), # putput: B x 8 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # B x 16 x 32 x 32
            nn.ReLU(), # B x 16 x 32 x 32
            nn.MaxPool2d(kernel_size=2, stride=2) # B x 16 x 16 x 16
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(32*16*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.output = nn.Linear(256, 40)

    # def forward function
    def forward(self, x):
        # feature extractor first
        x_post_fe = self.feature_extractor(x)
        # flatten
        x_flat = x_post_fe.view(x_post_fe.size(0), -1)
        # go through fcl
        x_fc = self.fully_connected_layer(x_flat)
        # get logits
        output = self.output(x_fc)
        # return logits
        return output

# create an instance of our model
model = SimpleCNN()
print(model)


### hyperparams ###
lr = 0.0001
epochs = 100
batch_size = 16



# do optimizer loss
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()


# define our training loop
def training_loop(dataloader, model, optimizer, loss_function):
    model.train()

    total_loss = 0
    # iteratre through dataloader
    for batch, (X, y) in enumerate(dataloader):
        # run X in model
        logits = model(X)
        # print(logits)
        # loss function
        loss = loss_function(logits, y)
        # print(loss)
        # refresh gradient
        optimizer.zero_grad()
        # backprop for model
        loss.backward()
        # changes weights and biases
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def testing_loop(dataloader, model, loss_function):
    # set model to eval mode
    model.eval()

    correct = 0
    total = 0
    total_loss = 0
    # iteratre through dataloader
    with torch.no_grad():
        for X, y in dataloader:
            # run X in model
            logits = model(X)
            # do a softmax
            pred = logits.argmax(1)
            loss = loss_function(logits, y)
            total_loss += loss.item()
            correct += (pred == y).sum().item()
            total += y.size(0)

    # print accuracy
    return 100 * correct / total, total_loss / len(dataloader)



# get losses
training_loss = []
testing_loss = []


for epoch in range(epochs):
    train_loss = training_loop(train_dataloader, model, optimizer, loss_function)
    test_acc, test_loss = testing_loop(test_dataloader, model, loss_function)
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # store loss
    training_loss.append(train_loss)
    testing_loss.append(test_loss)


# plot

plt.plot(training_loss)
plt.plot(testing_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")






######################################################################################

# import boston dataset


# load the dataset
bc = load_breast_cancer()

# prepare the data
X = bc.data
X.shape
y = bc.target
y.shape
len(np.unique(y))

# 569 samples, 30 features, binary


# create torch
X_tensor = torch.tensor(X, dtype=torch.float32)
X_tensor.shape
y_tensor = torch.tensor(y, dtype=torch.long)
y_tensor.shape


# make dataset

# make training and test splits
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# create a dataset and dataloader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# create a new model, MLP
class smallMLP(nn.Module):
    def __init__(self):
        super(smallMLP, self).__init__()

        self.linear_layer = nn.Sequential(
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(15),
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(10),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(5),
            nn.Linear(5,2)
        )

    def forward(self, x):
        # go through linear layer and get logits
        logits = self.linear_layer(x)
        # return logits
        return logits



model = smallMLP()
print(model)


### hyperparams ###
lr = 0.0001
epochs = 200
batch_size = 16


# do optimizer loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()


# define our training loop
def training_loop(dataloader, model, optimizer, loss_function):
    model.train()

    total_loss = 0
    # iteratre through dataloader
    for batch, (X, y) in enumerate(dataloader):
        # run X in model
        logits = model(X)
        # print(logits)
        # loss function
        loss = loss_function(logits, y)
        # print(loss)
        # refresh gradient
        optimizer.zero_grad()
        # backprop for model
        loss.backward()
        # changes weights and biases
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def testing_loop(dataloader, model, loss_function):
    # set model to eval mode
    model.eval()

    correct = 0
    total = 0
    total_loss = 0
    # iteratre through dataloader
    with torch.no_grad():
        for X, y in dataloader:
            # run X in model
            logits = model(X)
            # do a softmax
            pred = logits.argmax(1)
            loss = loss_function(logits, y)
            total_loss += loss.item()
            correct += (pred == y).sum().item()
            total += y.size(0)

    # print accuracy
    return 100 * correct / total, total_loss / len(dataloader)



# get losses
training_loss = []
testing_loss = []


for epoch in range(epochs):
    train_loss = training_loop(train_dataloader, model, optimizer, loss_function)
    test_acc, test_loss = testing_loop(test_dataloader, model, loss_function)
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # store loss
    training_loss.append(train_loss)
    testing_loss.append(test_loss)


# plot

plt.plot(training_loss, label = "training lss")
plt.plot(testing_loss, label = "testing loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()





###################################################################################



# load in mnist dataset
# load in mnist dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


next(iter(train_dataloader))[0].shape



# define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.feature_extractor = nn.Sequential( # B x 1 x 64 x 64
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # output: B x 8 x 64 x 64
            nn.ReLU(), # B x 8 x 64 x 64
            nn.AvgPool2d(kernel_size=2, stride=2), # putput: B x 8 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # B x 16 x 32 x 32
            nn.ReLU(), # B x 16 x 32 x 32
            nn.MaxPool2d(kernel_size=2, stride=2) # B x 16 x 16 x 16
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(32*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.output = nn.Linear(256, 10)

    # def forward function
    def forward(self, x):
        # feature extractor first
        x_post_fe = self.feature_extractor(x)
        # flatten
        x_flat = x_post_fe.view(x_post_fe.size(0), -1)
        # go through fcl
        x_fc = self.fully_connected_layer(x_flat)
        # get logits
        output = self.output(x_fc)
        # return logits
        return output

# create an instance of our model
model = SimpleCNN()
print(model)


### hyperparams ###
lr = 0.01
epochs = 10
batch_size = 64



# do optimizer loss
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()


# define our training loop
def training_loop(dataloader, model, optimizer, loss_function):
    model.train()

    total_loss = 0
    # iteratre through dataloader
    for batch, (X, y) in enumerate(dataloader):
        # run X in model
        logits = model(X)
        # print(logits)
        # loss function
        loss = loss_function(logits, y)
        # print(loss)
        # refresh gradient
        optimizer.zero_grad()
        # backprop for model
        loss.backward()
        # changes weights and biases
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def testing_loop(dataloader, model, loss_function):
    # set model to eval mode
    model.eval()

    correct = 0
    total = 0
    total_loss = 0
    # iteratre through dataloader
    with torch.no_grad():
        for X, y in dataloader:
            # run X in model
            logits = model(X)
            # do a softmax
            pred = logits.argmax(1)
            loss = loss_function(logits, y)
            total_loss += loss.item()
            correct += (pred == y).sum().item()
            total += y.size(0)

    # print accuracy
    return 100 * correct / total, total_loss / len(dataloader)



# get losses
training_loss = []
testing_loss = []


for epoch in range(epochs):
    train_loss = training_loop(train_dataloader, model, optimizer, loss_function)
    test_acc, test_loss = testing_loop(test_dataloader, model, loss_function)
    # if epoch % 5 == 0 or epoch == epochs - 1:
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # store loss
    training_loss.append(train_loss)
    testing_loss.append(test_loss)


# plot

plt.plot(training_loss)
plt.plot(testing_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")



