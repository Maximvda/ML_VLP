import torch.nn as nn
import numpy as np

#Just print the model to see the network layers
class model(nn.Module):
    def __init__(self, size, output_nc, model_type, nf, extra_layers, use_sigmoid=True):
        super(model, self).__init__()
        if 'FC_expand' in model_type:
            self.main = fc(size,output_nc, nf, extra_layers, True)
        else:
            self.main = fc(size,output_nc, nf, extra_layers, False)

    def forward(self, input):
        return self.main(input)

class fc(nn.Module):
    def __init__(self, size,output_nc, nf, extra_layers, expand):
        super(fc, self).__init__()
        nb_layers = extra_layers + 1
        f_mult = 1
        prev_f_mult = 1

        submodule = fc_layer(size, nf)
        for i in range(nb_layers):
            prev_f_mult = f_mult
            f_mult = 2**(int(np.log2(prev_f_mult)-1)) if (i >= nb_layers/2 and expand) else 2**(i+1)
            submodule = fc_layer(nf*prev_f_mult, nf*f_mult, submodule=submodule)

        self.main = fc_layer(nf*f_mult, output_nc, final=True, submodule=submodule)

    def forward(self, input):
        return self.main(input)

class fc_layer(nn.Module):
    def __init__(self,input_nc, output_nc, final=False, submodule=None):
        super(fc_layer, self).__init__()
        lin = nn.Linear(input_nc, output_nc)
        relu = nn.LeakyReLU(0.2, True)
        #dropout = nn.Dropout(0.5)
        layer = [lin, relu]
        if submodule == None:
            model = layer
        elif final:
            model = [submodule] + [lin, nn.Tanh()]
        else:
            model = [submodule] + layer
        self.main = nn.Sequential(*model)

    def forward(self, x):
        return self.main(x)
