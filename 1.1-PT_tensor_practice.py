#TENSOR MATH & COMPARISON OPERATIONS WITH PYTORCH

import torch 

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition 
z1 = torch.empty(3)
torch.add(x,y,out = z1)
print(z1)

z2 = torch.add(x,y)
z = x + y 

# Subtraction

z = x - y

# Division

z = torch.true_divide(x,y) # If x and y are equal shaped, then it will do element wise division
                           # else it will broadcast

# inplace operations (it doesnt create a copy so it is faster)
t = torch.zeros(3) 
t.add_(x) # Any operation followed by
t += x #Inplace Addition. t = t + x is not inplace

#Exponentiation

z = x.pow(2) # Element wise exponentiation
z = x ** 2 # Element wise exponentiation
print(z)

#Simple comparision

z = x > 0  #Element wise comparison
z = x < 0 #Element wise comparison
print(z)

# Matrix multiplication

x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) # Outputshape = 2 x 3
x3 = x1.mm(x2) # Another way to do that

#matrix exponentiation

matrix_exp = torch.rand(5,5)
print(matrix_exp)
print(matrix_exp.matrix_power(3))  #Raise the entire matrix to that power

# element wise multiplication

z = x * y 
print(z)

# dot product
z = torch.dot(x,y)
print(z)

# batch matrix multiplication

batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m)) 
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) #output shape = (batch, n, p)

# Example of broadcasting

x1 = torch.rand((5,5))
x2 = torch.rand((1,5)) #This matrix will expand which is known as broadcast

z = x1 - x2 
z = x1 ** x2

#other useful tensor application

sum_x = torch.sum(x,dim = 0) #sum
values, indices = torch.max(x, dim = 0) #max value
values, indices = torch.min(x, dim = 0) #min value
abs_x  =torch.abs(x) #Absolute value 
z = torch.argmax(x,dim=0) #returns index of maximum value
z = torch.argmin(x,dim=0) #returns index of minimum value
mean_x = torch.mean(x.float(), dim = 0) #returns mean value, float is required
z = torch.eq(x,y) #returns a matrix of true and false values based on their equality
torch.sort(y, dim = 0, descending = False) # Sorts the values

z = torch.clamp(x, min = 0, max = 10) #makes all values less than 0 to 0. every value greater than max to max 

x = torch.tensor([1,0,1,1,1], dtype = torch.bool)
z = torch.any(x) #returns true if any of the value is true
print(z)
z = torch.all(x) #returns true if all of the value are True
print(z)

