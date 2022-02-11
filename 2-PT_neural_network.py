# Imports
# Create fully connected network
# Set device
# Hyperparameters
# Load Data
# Initialize network
# Loss and Optimizer
# Train Network
# Check accuracy on training and test to see how good is our model

# =========================================================================================================

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
# functions without parameters. such as Relu, tanh etc
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected network


class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # MNIST HAS 28x28 images = 784 nodes
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model = NN(784,10) #For test
# x = torch.randn(64, 784)
# print(model(x).shape) #torch.Size([64, 10])


# Set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 1

# Load Data
train_dataset = datasets.MNIST(root='datasets/', train = True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root='datasets/', train = False, transform = transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = True)


# Initialize network
model = NN(input_size = input_size, num_classes = num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        #here, data.shape = [64,1,28,28]
        # Get to correct shape [64, 784]
        data = data.reshape(data.shape[0], -1) #by -1 , it will flatten 1x28x28 to 784

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam steps
        optimizer.step()

# Check accuracy on training and test to see how good is our model

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
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
        
        print("Got {}/{} with accurary {}".format(num_correct,num_samples,num_correct/num_samples*100))

        model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

# Output 
# Checking accuracy on training data
# Got 55959/60000 with accurary 93.265
# Checking accuracy on test data
# Got 9323/10000 with accurary 93.23