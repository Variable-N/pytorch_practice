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

z = x - yield

# Division

z = torch.true_divide(x,y) # If x and y are equal shaped, then it will do element wise division
                           # else it will broadcast

# inplace operations 
t = torch.zeros(3) 
t.add_(x) # Any operation followed by
