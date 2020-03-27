import torch.nn as nn
import numpy as np

#Setup the correct model depending on model type and model parameters
#As defined in the readme file
class Model(nn.Module):
    def __init__(self, input_size, output_size, model_type, nf, hidden_layers):
        super(Model, self).__init__()
        if 'Type_1' == model_type:
            self.main = Fc(input_size, output_size, nf, hidden_layers, False)
        elif 'Type_2' == model_type:
            self.main = Fc(input_size, output_size, nf, hidden_layers, True)
        else:
            print('Model type is not implemented.')

    def forward(self, input):
        return self.main(input)

class Fc(nn.Module):
    def __init__(self, input_size, output_size, nf, hidden_layers, Type_2):
        super(Fc, self).__init__()
        f_mult = 1
        prev_f_mult = 1

        submodule = Fc_layer(input_size, nf)
        for i in range(hidden_layers-1):
            prev_f_mult = f_mult
            f_mult = 2**(int(np.log2(prev_f_mult)-1)) if (i >= np.floor((hidden_layers)/2) and Type_2) else 2**(i+1)
            submodule = Fc_layer(nf*prev_f_mult, nf*f_mult, submodule=submodule)

        self.main = Fc_layer(nf*f_mult, output_size, final=True, submodule=submodule)

    def forward(self, input):
        return self.main(input)

#Fully connected layer with input_f input features and output_f output features
#If submodule is given then layer is appended after it
#If it is final layer then sigmoid is added at the end
class Fc_layer(nn.Module):
    def __init__(self,input_f, output_f, final=False, submodule=None):
        super(Fc_layer, self).__init__()
        lin = nn.Linear(input_f, output_f)
        relu = nn.LeakyReLU(0.2, True)
        layer = [lin, relu]
        if submodule == None:
            model = layer
        elif final:
            model = [submodule] + [lin, nn.Sigmoid()]
        else:
            model = [submodule] + layer
        self.main = nn.Sequential(*model)

    def forward(self, x):
        return self.main(x)
