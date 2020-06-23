import numpy as np
import torch
import math
from torch.utils.data.dataset import Dataset


class AutoEncoderDataset(Dataset):
    '''Data for Dataloader
    '''
    def __init__(self, X):
        self.data = X.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

def load_data(path):
    '''Load the data
    '''
    data = np.genfromtxt(path)
    return data[:,1:]

def calculate_fan_in_and_fan_out(tensor):
    '''Return fan_in and fan_out of a tensor
    '''
    N_input = tensor.size(1)
    N_output = tensor.size(0)
    receptive_field = tensor[0][0].numel()
    fan_in = N_input * receptive_field
    fan_out = N_output * receptive_field

    return fan_in, fan_out

def custom_xavier_uniform_(tensor, gain=1.):
    '''Return Uniform within [-U,U]
    with U = \sqrt{\frac{6}{1+fan_in+fan_out}}
    '''
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    U = gain * math.sqrt(6.0 / float(1 + fan_in + fan_out))
    with torch.no_grad():
        return tensor.uniform_(-U, U)

def init_weights(m):
    '''Initialize weights
    '''
    if isinstance(m, torch.nn.Linear):
        custom_xavier_uniform_(m.weight, gain=1.) #initialize weights

def init_weights_bias(m): 
    '''Initialize weights and biases
    '''
    if isinstance(m, torch.nn.Linear):
        custom_xavier_uniform_(m.weight, gain=1.) #initialize weights
        if m.bias is not None: ##initialize bias
            fan_in, fan_out = calculate_fan_in_and_fan_out(m.weight)
            U = 1. * math.sqrt(6.0 / float(1 + fan_in + fan_out))
            torch.nn.init.uniform_(m.bias, -U, U)