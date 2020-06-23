import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
from utils import *

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Found", device, "device")


class FunctionalTiedAutoEncoder(nn.Module):
    def __init__(self, d_):
        super().__init__()
        self.encoder = nn.Linear(256, d_, bias=False)
        self.tanh = nn.Tanh()
        self.bias1 = nn.Parameter(torch.zeros(d_), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(256), requires_grad=True)
        fan_in0, fan_out1 = calculate_fan_in_and_fan_out(self.encoder.weight)
        fan_in1, fan_out2 = calculate_fan_in_and_fan_out(
            self.encoder.weight.t())

        def U(fan_in, fan_out): return 1. * math.sqrt(
            6.0 / float(1 + fan_in + fan_out))
        torch.nn.init.uniform_(self.bias1.detach(), -U(fan_in0, fan_out1), U(fan_in0, fan_out1))
        torch.nn.init.uniform_(self.bias1.detach(), -U(fan_in1, fan_out2), U(fan_in1, fan_out2))

    def forward(self, inputs):
        encoded = self.encoder(inputs) + self.bias1
        encoded = self.tanh(encoded)
        reconstructed = F.linear(
            encoded, self.encoder.weight.t()) + self.bias2
        return reconstructed


if __name__ == '__main__':

    data1 = load_data('./data/zip.train.txt')
    train_data = AutoEncoderDataset(data1)
    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=7291, shuffle=False)

    data2 = load_data('./data/zip.test.txt')
    test_data = AutoEncoderDataset(data2)
    test_loader = Data.DataLoader(
        dataset=test_data, batch_size=2007, shuffle=False)


    LR = 0.1
    num_epochs = 5000

    d_ = range(1, 8)
    Eout = []
    for i in d_:
        k = 2**i
        model = FunctionalTiedAutoEncoder(d_=k)
        model.to(device)
        model.apply(init_weights)

        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            for data in train_loader:
                inputs = data.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, inputs)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        param -= LR * param.grad
            if (epoch+1) % 250 == 0:
                print('d_ = {}, epoch [{}/{}], loss:{:.4f}'
                      .format(k, epoch + 1, num_epochs, loss.item()))

        for data in test_loader:
            inputs = data.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            print("Loss on testing set %.4f"%loss)
        Eout.append(loss.item())

    print('Eout =', Eout)
    plt.style.use('ggplot')
    plt.plot(d_, Eout)
    plt.xlabel("$log_{2}\widetilde{d}$")
    plt.ylabel("$Eout(g)$")
    plt.title("Eout with contraint $w_{ij}^{(1)} = w_{ji}^{(2)}$")
    plt.savefig("./14.pdf", format="pdf")
    plt.show()