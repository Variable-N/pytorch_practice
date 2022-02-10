#PYTORCH TENSOR BASICS
import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype = torch.float32, device = device)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)  #torch.Size([2, 3]) 2 is row 3 is column
print(my_tensor.requires_grad)

#Other common initialization methods

x = torch.empty(size = (3,3)) #uninitialized values.
print(x)
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3))
print(x)
x = torch.ones((3,3))
print(x)
x = torch.eye(5,5)  # Creates an Identity matrix
print(x)
x = torch.arange(start = 0, end = 5, step = 1) #its like a range function
print(x)  #tensor([0, 1, 2, 3, 4])
x = torch.linspace(start = 0.1, end = 1, steps = 10) #from 0.1 to 1, we will get 10 values in the tensor
print(x)  #tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000])
x = torch.empty(size = (1,5)).normal_(mean = 0, std = 1)
print(x)  # Normal distribution 
x = torch.empty(size = (1,5)).uniform_(0,1)
print(x)  # Uniform distribution 
x = torch.diag(torch.ones(3)) #It creates 3 by 3 diagonal matrix
print(x) 

# How to initialize and covert tensor to other types (Int, float, double)

tensor = torch.arange(4)

print(tensor.bool()) #Boolean lol
print(tensor.short()) #int16
print(tensor.long()) #int64 *most used
print(tensor.half()) #float16
print(tensor.float()) #float32 *often used
print(tensor.double()) #float64

# Array to tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()