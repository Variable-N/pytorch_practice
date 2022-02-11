# Tensor Indexing

import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

print(x[0]) # x[0,:]
print(x[0].shape) 

print(x[:,0].shape)

print(x[2, 0:10]) #2nd example, features 0 to 9 [0,1,2, ...., 9]

x[0, 0] = 100

#Fancy Indexing 

x = torch.arange(10)
indices = [2,5,8]
print(x[indices]) 

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])

# More advanced indexing
x = torch.arange(10)
print(x[ (x < 2) | (x > 8)] )
print(x[ (x < 2) & (x > 8)] )
print(x [x.remainder(2) == 0])

# Useful OPERATIONS

print(torch.where(x > 5, x, x*2)) # if the value is greater than 5 then it will set x to x otherwise x*2
print(torch.tensor([0,0,1,2,2,3,4]).unique()) #Returns all the unique values
print(x.ndimension()) # Returns the number of dimensions
print(x.numel()) # returns the number of elements