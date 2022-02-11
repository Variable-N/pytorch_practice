# Tensor Reshaping

import torch

x = torch.arange(9)

x_3x3 = x.view(3,3) # tensor stores contigously in memory
x_3x3 = x.reshape(3,3) # tensor stores incontigously in memory. safe to use but can cause performance loss
print(x_3x3)
print(x_3x3.shape)

y = x_3x3.t()  # returns transpose of the tensor
print(y.reshape(9)) # y.view(9) will be error due to contigous related errors.
print(y.contiguous().view(9)) # by making it contigous then we can use view  ==== JUST USE RESHAPE :)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))

print(torch.cat((x1,x2), dim = 0))  #Dim = 0 means row wise concatination
print(torch.cat((x1,x2), dim = 0).shape)  #torch.Size([4, 5])
print(torch.cat((x1,x2), dim = 1))  #Dim = 1 means column wise concatination
print(torch.cat((x1,x2), dim = 1).shape)  #torch.Size([2, 10])

#To unenroll the 2,5 to 1,10 
z = x1.view(-1)
print(z.shape)


batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1) # It will remain the batch untouched, but flatten the 2,5 
print(z.shape)

z = x.permute(0, 2, 1) # this will change (batch, 2, 5) to (batch, 5, 2). changes the dimensions 
print(z.shape)

x = torch.arange(10) # [10]
print(x.unsqueeze(0).shape) # this will make [10] to [1,10]
print(x.unsqueeze(1).shape) # this will make [10] to [10,1]

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # [1x1x10]
z = x.squeeze(1)
print(z.shape)