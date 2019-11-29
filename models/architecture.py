import torch.nn as nn
import numpy as np

#No comment just print the model to see its layers
class cnn(nn.Module):
    def __init__(self, size, nc, nf, extra_layers, use_sigmoid=True):
        super(cnn, self).__init__()
        num_downs = int(min(size[0],size[1])-2)
        submodule = None
        for i in range(0,extra_layers):
            if i == 0:
                submodule = DownConv(nc,nf,kernel=3)
                prev_f_mult = 1
                f_mult = 1
            else:
                prev_f_mult = f_mult
                f_mult = min(2 ** i, 8)
                submodule = DownConv(nf*prev_f_mult,nf*f_mult, kernel=3, submodule=submodule)

        if submodule is None:
            submodule = DownConv(nc, nf)
            prev_f_mult = 1
            f_mult = 1

        for i in range(1,num_downs):
            prev_f_mult = f_mult
            f_mult = min(2 ** i, 8)
            submodule = DownConv(nf*prev_f_mult,nf*f_mult, submodule=submodule)

        submodule = DownConv(nf*f_mult,2, final=True, submodule=submodule)

        if use_sigmoid:
            sigmoid = nn.Sigmoid()
            model = [submodule] + [sigmoid]

        self.main = nn.Sequential(*model)

    def forward(self, input):
        #output = self.conv1(input)
        #print(output.size())
        return self.main(input)

class DownConv(nn.Module):
    def __init__(self,input_nc, output_nc, kernel=4, final=False, submodule=None):
        super(DownConv, self).__init__()
        down_conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel, stride=1, padding=1, bias=False)
        down_norm = nn.BatchNorm2d(output_nc)
        down_relu = nn.LeakyReLU(0.2, True)
        dropout = nn.Dropout(0.5)
        down = [down_conv, down_norm, down_relu]
        if submodule == None:
            model = [down_conv, down_relu]
        elif final:
            model = [submodule] + [dropout, down_conv]
        else:
            model = [submodule] + down
        self.main = nn.Sequential(*model)

    def forward(self, x):
        return self.main(x)
