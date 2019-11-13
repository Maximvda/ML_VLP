import torch.nn as nn
import numpy as np

class cnn(nn.Module):
    def __init__(self, size, nc, nf, use_sigmoid=True):
        super(cnn, self).__init__()
        #num_downs = int(np.log2(min(size[0],size[1]))-1)

        #submodule = DownConv(nc, nf)
        #submodule = DownConv(nf, nf*2, submodule)
        #submodule = DownConv(nf*2, nf*4, submodule)
        #model = DownConv(nf*4, 1, final=True, submodule=submodule)
        self.conv1 = DownConv(nc, nf)
        self.conv2 = DownConv(nf, nf*2)
        self.conv3 = DownConv(nf*2, nf*4)
        self.conv4 = DownConv(nf*4, nf*8)
        self.dropout = nn.Dropout(0.5)
        self.conv5 = DownConv(nf*8, 2,final=True)
        self.use_sigmoid = use_sigmoid

        #prev_f_mult = 1
        #f_mult = 1
        #i=1
        #for i in range(1,num_downs):
        #    prev_f_mult = f_mult
        #    f_mult = min(2 ** i, 8)
        #    submodule = DownConv(nf*prev_f_mult,nf*f_mult, submodule)

        #prev_f_mult = f_mult
        #f_mult = min(2 ** i, 8)
        #submodule = DownConv(nf*prev_f_mult,nf*f_mult, submodule)

        #down = nn.Conv2d(nf*f_mult, 1, 4, 1, 1, bias=False)
        #down = nn.Conv2d(ndf*f_mult, 1, 4, 1, [1,0], bias=False)
        #model = [submodule] + [down]
        #if use_sigmoid:
        #    sigmoid = nn.Sigmoid()
        #    model = [model] + [sigmoid]

        #self.main = nn.Sequential(*model)

    def forward(self, input):
        #print("In forward pass")
        output = self.conv1(input)
        #print(output.size())
        output = self.conv2(output)
        #print(output.size())
        output = self.conv3(output)
        #print(output.size())
        output = self.conv4(output)
        #print(output.size())
        output = self.dropout(output)
        output = self.conv5(output)
        if self.use_sigmoid:
            sig = nn.Sigmoid()
            output = sig(output)
        #print(output.size())
        return output

class DownConv(nn.Module):
    def __init__(self,input_nc, output_nc, final=False, submodule=None):
        super(DownConv, self).__init__()
        #str = [4, 2] if final else 2
        down_conv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=1, padding=1, bias=False)
        down_norm = nn.BatchNorm2d(output_nc)
        down_relu = nn.LeakyReLU(0.2, True)
        down = [down_conv, down_norm, down_relu]
        if submodule == None:
            model = [down_conv, down_relu]
        elif final:
            model = [submodule] + [down_conv]
        else:
            model = [submodule] + down
        self.main = nn.Sequential(*model)

    def forward(self, x):
        return self.main(x)
