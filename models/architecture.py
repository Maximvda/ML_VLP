import torch.nn as nn
import numpy as np

#Just print the model to see the network layers
class model(nn.Module):
    def __init__(self, size, model_type, nf, extra_layers, use_sigmoid=True):
        super(model, self).__init__()
        if 'CNN' in model_type:
            self.main = cnn(size, nf, extra_layers, use_sigmoid=True)
        elif 'FC_expand' in model_type:
            self.main = fc(size, nf, extra_layers, True)
        else:
            self.main = fc(size, nf, extra_layers, False)
        print(self.main)

    def forward(self, input):
        return self.main(input)

class fc(nn.Module):
    def __init__(self, size, nf, extra_layers, expand):
        super(fc, self).__init__()
        nb_layers = extra_layers + 1
        f_mult = 1
        prev_f_mult = 1

        submodule = fc_layer(size, nf)
        for i in range(nb_layers):
            prev_f_mult = f_mult
            f_mult = 2**(int(np.log2(prev_f_mult)-1)) if (i >= nb_layers/2 and expand) else 2**(i+1)
            submodule = fc_layer(nf*prev_f_mult, nf*f_mult, submodule=submodule)

        self.main = fc_layer(nf*f_mult, 3, final=True, submodule=submodule)

    def forward(self, input):
        #return self.main(input)[:,0,:]
        return self.main(input)

class cnn(nn.Module):
    def __init__(self, size, nf, extra_layers, use_sigmoid=True):
        super(cnn, self).__init__()
        shape = int(np.ceil(np.sqrt(size)))
        num_downs = int(shape-2)
        submodule = None
        for i in range(0,extra_layers):
            if i == 0:
                submodule = DownConv(1,nf,kernel=3)
                prev_f_mult = 1
                f_mult = 1
            else:
                prev_f_mult = f_mult
                f_mult = min(2 ** i, 8)
                submodule = DownConv(nf*prev_f_mult,nf*f_mult, kernel=3, submodule=submodule)

        if submodule is None:
            submodule = DownConv(1, nf)
            prev_f_mult = 1
            f_mult = 1

        #If depth of model would be to large downsample faster (division of size by 2)
        while num_downs > 3:
            prev_f_mult = f_mult
            f_mult = min(2**(int(np.log2(prev_f_mult)+1)), 8)
            submodule = DownConv(nf*prev_f_mult,nf*f_mult, stride=2, submodule=submodule)
            shape = shape/2
            num_downs = int(shape-2)

        for i in range(1,num_downs):
            prev_f_mult = f_mult
            f_mult = min(2 ** i, 8)
            submodule = DownConv(nf*prev_f_mult,nf*f_mult, submodule=submodule)

        submodule = DownConv(nf*f_mult,3, final=True, submodule=submodule)

        if use_sigmoid:
            sigmoid = nn.Sigmoid()
            model = [submodule] + [sigmoid]

        self.main = nn.Sequential(*model)

    def forward(self, input):
        return self.main(input)[:,:,0,0]

class fc_layer(nn.Module):
    def __init__(self,input_nc, output_nc, final=False, submodule=None):
        super(fc_layer, self).__init__()
        lin = nn.Linear(input_nc, output_nc)
        relu = nn.LeakyReLU(0.2, True)
        #dropout = nn.Dropout(0.5)
        norm = nn.BatchNorm2d(output_nc)
        layer = [lin, norm, relu]
        if submodule == None:
            model = layer
        elif final:
            #model = [submodule] + [dropout, lin, nn.Sigmoid()]
            model = [submodule] + [lin, nn.Sigmoid()]
        else:
            model = [submodule] + layer
        self.main = nn.Sequential(*model)

    def forward(self, x):
        return self.main(x)

class DownConv(nn.Module):
    def __init__(self,input_nc, output_nc, kernel=4, stride=1, final=False, submodule=None):
        super(DownConv, self).__init__()
        down_conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel, stride=stride, padding=1, bias=False)
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
