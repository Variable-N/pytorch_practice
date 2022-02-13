# Imports
# Set device
# Hyperparameters
# Create a Bidirectional LSTM RNN https://www.youtube.com/watch?v=D-a6dwXzJ6s&ab_channel=KrishNaik
# Load Data
# Initialize network
# Loss and Optimizer
# Train Network
# Check accuracy on training and test to see how good is our model

# =========================================================================================================

# Imports
import torch
import torchvision
import torch.nn as nn # All Neural Netowrk Modules, nn.Lineaer, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # Fir all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # functions without parameters. such as Relu, tanh etc
from torch.utils.data import DataLoader #Gives easier dataset management and create mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms #Transformations we can perform on our dataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 3

# Create a Bidirectional LSTM

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))  # it returns hidden state and cell states but we dont need them
        out = self.fc(out[:, -1, :])

        return out

# Load Data
train_dataset = datasets.MNIST(root='datasets/', train = True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root='datasets/', train = False, transform = transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = True)


# Initialize network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device = device).squeeze(1)     # MNIST has Nx1x28x28 but RNN expects Nx28x28. so Squeeze(1) will remove the 1.
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
            x = x.to(device = device).squeeze(1)
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
# Got 58806/60000 with accurary 98.01000213623047
# Checking accuracy on test data
# Got 9757/10000 with accurary 97.56999969482422