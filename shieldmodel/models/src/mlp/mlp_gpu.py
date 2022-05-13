import torch
import torch.nn as nn
import torch.nn.functional as F
import socket

class myNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim = 1)
        return out


device_gpu=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu=torch.device("cpu")
x = torch.rand((1, 1, 28, 28)).to(device_gpu)
model = myNN().to(device_gpu)
for i in range(3000):
    y = model.forward(x)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
total_time = 0
for rep in range(1000):
    starter.record()
    _ = model(x)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    total_time += curr_time
mean_syn = total_time / 1000
print("Time elapsed: "+str(mean_syn))


    