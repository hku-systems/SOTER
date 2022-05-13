import torch
import torch.nn as nn
import torch.nn.functional as F

class myNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.c0 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.c1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.c4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.c8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.c9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.c10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.c11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.c12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.c13 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.c14 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.f0 = nn.Linear(25088, 4096)
        self.f1 = nn.Linear(4096, 4096)
        self.f2 = nn.Linear(4096, 1000)
        self.mxp2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.c(x)
        out = F.relu(out)
        out = self.c0(out)
        out = F.relu(out)
        out = self.mxp2d(out)
        out = self.c1(out)
        out = F.relu(out)
        out = self.c2(out)
        out = F.relu(out)
        out = self.mxp2d(out)
        out = self.c3(out)
        out = F.relu(out)
        out = self.c4(out)
        out = F.relu(out)
        out = self.c5(out)
        out = F.relu(out)
        out = self.c6(out)
        out = F.relu(out)
        out = self.mxp2d(out)
        out = self.c7(out)
        out = F.relu(out)
        out = self.c8(out)
        out = F.relu(out)
        out = self.c9(out)
        out = F.relu(out)
        out = self.c10(out)
        out = F.relu(out)
        out = self.mxp2d(out)
        out = self.c11(out)
        out = F.relu(out)
        out = self.c12(out)
        out = F.relu(out)
        out = self.c13(out)
        out = F.relu(out)
        out = self.c14(out)
        out = F.relu(out)
        out = self.mxp2d(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.f0(out)
        out = F.relu(out)
        out = self.f1(out)
        out = F.relu(out)
        out = self.f2(out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device_gpu=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu=torch.device("cpu")
x = torch.rand((1, 3, 224, 224)).to(device_gpu)
model = myNN().to(device_gpu)

# warmup
for i in range(3000):
    _ = model.forward(x)

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
print("Time per inference: "+str(mean_syn))