import torch.nn as nn
import numpy as np

class cnn(nn.Module):
    def __init__(self, size, nc, nf, use_sigmoid=False):
        super(cnn, self).__init__()
        num_downs = int(np.log2(min(size[0],size[1]))-1)

        submodule = DownConv(nc, nf)
        self.conv1 = nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1, bias=False)

        prev_f_mult = 1
        f_mult = 1
        i=1
        for i in range(1,num_downs):
            prev_f_mult = f_mult
            f_mult = min(2 ** i, 8)
            submodule = DownConv(nf*prev_f_mult,nf*f_mult, submodule)

        prev_f_mult = f_mult
        f_mult = min(2 ** i, 8)
        submodule = DownConv(nf*prev_f_mult,nf*f_mult, submodule)

        down = nn.Conv2d(nf*f_mult, 1, 4, 1, 1, bias=False)
        #down = nn.Conv2d(ndf*f_mult, 1, 4, 1, [1,0], bias=False)
        model = [submodule] + [down]
        if use_sigmoid:
            sigmoid = nn.Sigmoid()
            model = [model] + [sigmoid]

        self.main = nn.Sequential(*model)

    def forward(self, input):
        print("In forward pass")
        print(type(input))
        output = self.conv1(input)
        print(output)
        return self.main(input)

class DownConv(nn.Module):
    def __init__(self,input_nc, output_nc, submodule=None):
        super(DownConv, self).__init__()
        down_conv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
        down_norm = nn.BatchNorm2d(output_nc)
        down_relu = nn.LeakyReLU(0.2, True)
        down = [down_conv, down_norm, down_relu]
        if submodule == None:
            model = [down_conv, down_relu]
        else:
            model = [submodule] + down
        self.main = nn.Sequential(*model)

    def forward(self, x):
        return self.main(x)
