import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from utils import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Found", device, "device")


class AutoEncoder(nn.Module):
    def __init__(self, d_):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(256, d_)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(d_, 256)

    def forward(self, x):
        x = self.encoder(x)
        x = self.tanh(x)
        x = self.decoder(x)
        return x

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
        model = AutoEncoder(d_=k)
        model.to(device)
        model.apply(init_weights_bias)

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
    plt.title("Eout without contraint $w_{ij}^{(1)} = w_{ji}^{(2)}$")
    plt.savefig("./12.pdf", format="pdf")
    plt.show()