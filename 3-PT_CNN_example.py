# Imports
# Create a Simple CNN
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
import torch.nn as nn # All Neural Netowrk Modules, nn.Lineaer, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # Fir all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # functions without parameters. such as Relu, tanh etc
from torch.utils.data import DataLoader #Gives easier dataset management and create mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms #Transformations we can perform on our dataset

# TODO: CREATE SIMPLE CNN
# Conv2d output feature,
#                          n_out = [ (n_in + 2p - 1)/ s] + 1
#Here,
#      n_in = number of input features            k = convolution kernel size
#      n_out = number of output features          p = convolution padding size    s = convolution stride size
#
#      Same convolution means n_in = n_out

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels = 8, kernel_size= (3,3), stride=(1,1), padding=(1,1)) #Same convulation
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

# FOR TEST
# model = CNN()
# x = torch.randn(64,1,28,28)
# print(x.shape)
# print(model(x).shape)
# exit()

# Set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 5

# Load Data
train_dataset = datasets.MNIST(root='datasets/', train = True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root='datasets/', train = False, transform = transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = True)


# Initialize network
model = CNN().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)
        

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
# Got 59242/60000 with accurary 98.73666381835938
# Checking accuracy on test data
# Got 9839/10000 with accurary 98.3899917602539