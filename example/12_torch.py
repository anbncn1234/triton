import torch

import numpy as np

x = torch.linspace(0, 100, steps=100, dtype=torch.float32)
print(x)

rand = torch.rand(100)*10
y = x + rand
print(y)
import matplotlib.pyplot as plt 

'''
plt.figure(figsize=(10,8))
plt.plot(x.numpy(), y.numpy(), 'b.')  # ?????
plt.xlabel('x')  # x ???
plt.ylabel('y')  # y ???
plt.title('y = x + noise')  # ??
plt.grid(True)  # ????
plt.show()  # ????
'''


a = torch.rand(1, requires_grad = True)
b = torch.rand(1, requires_grad = True)

print(a,b)
lr = 0.001

#print(fx_)

for i in range(1):
    fx_real = y
    #print(fx_real) 
    fx_ = (a * x + b)
    print(fx_)
    loss = (fx_real - fx_)
    print(loss)
    print(loss.pow(2))
    L = loss.pow(2).sum()
    print(L)
    #loss_a = x * (fx_real - fx_)
    #loss_b =  fx_real - fx_
    #a = a - lr* loss_a
    #b = b - lr* loss_b
    
print(a,b)