import torch
'''
x=torch.empty(5,3)
print(x)

#construct a matrix filled zeros and dtype long
x=torch.zeros(5,3,dtype=torch.long)
print(x)
print(x.size())

#resize/reshape tensor
x=torch.randn(4,4)
y=x.view(16)
z=x.view(2,8)
print(x,'\n', y,'\n',z)


#converting a torch tensor to a numpy array
a=torch.ones(5)
print(a)
b=a.numpy()
print(b)
'''


#define the network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #1 input image channel,6 output channels,3*3 square
        # convolution kernel
