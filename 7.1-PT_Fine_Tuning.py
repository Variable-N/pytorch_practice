#Imports

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

#Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters

in_channels = 3 
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epoch = 5

#Load pretrain model & modify it

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x

model = torchvision.models.vgg16(pretrained = True)

for param in model.parameters():
    param.requires_grad = False 
    
model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(512,100),
    nn.ReLU(),
    nn.Linear(100,10)
)
model.to(device = device)
print(model)


# Load Data
train_dataset = datasets.CIFAR10(root='datasets1/', train = True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Train Network

for epoch in range(num_epoch):
    losses = []

    for batch_idx, (data,targets) in enumerate(train_loader):

        #Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)

        #Forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        #backward 
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam steps
        optimizer.step()
    
    mean_loss = sum(losses)/len(losses)
    print("Cost at epoch {} was {}".format(epoch, mean_loss))



def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
        
        print("Got {}/{} with accuracy {}.".format(num_correct,num_samples,num_correct/num_samples*100))

        model.train()

check_accuracy(train_loader,model)