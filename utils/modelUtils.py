import torch
import os
import numpy as np
from trainers.model import Model
import torch.optim as optim

from utils.config import get_PBT_choices

#Loads the entire state of the training process if any
#Else it initialises everything to start training
#When reload_model is set to true it loads the model and model parameters from the file
def setup_model(self, file, reload_model=False):
    resultpath = os.path.join(self.result_root,file)
    #check if there exists a checkpoint if so load it
    if os.path.isfile(resultpath):
        checkpoint = torch.load(resultpath, map_location=self.device)
        #Reloads model parameters from the checkpoint
        if reload_model:
            self.size = checkpoint['size']
            self.TX_config = checkpoint['TX_config']; self.TX_input = checkpoint['TX_input']
            self.blockage = checkpoint['blockage']; self.output_nf = checkpoint['output_nf']
            self.model_type = checkpoint['model_type']
            self.nf = checkpoint['nf']
            self.hidden_layers = checkpoint['hidden_layers']
            self.model = Model(self.size, self.output_nf, self.model_type,
                            self.nf, self.hidden_layers)
            for param_group in checkpoint['optim']['param_groups']:
                lr = param_group['lr']
                betas = param_group['betas']
            self.optim = optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        #Model is setup with defined parameters
        #When model parameters are not loaded from checkpoint
        else:
            self.model = Model(self.size, self.output_nf, self.model_type, self.nf, self.hidden_layers)
            self.optim = optim.Adam(self.model.parameters(), self.learning_rate, betas=(0.5, 0.999))

        #Load state variables from checkpoint
        if not checkpoint['model'] == None:
            self.model.load_state_dict(checkpoint['model'])
            self.optim.load_state_dict(checkpoint['optim'])
        self.iter = checkpoint['iter']
        self.learning = checkpoint['learning']
        self.min_dist = checkpoint['min_dist']
        self.best_iter = checkpoint['best_iter']
        self.batch_size = checkpoint['batch_size']
        #Correctly push all values from optimiser to device
        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    #Init model if no checkpoint
    else:
        init_model(self, file, reload_model)
    #Push model to device
    self.model.to(self.device)

def init_model(self, file, reload_model):
    self.iter = 0; self.best_iter = 0; self.min_dist = {'2D': np.inf, 'z': np.inf, '3D': np.inf}
    if reload_model:
        #Get the random choices for PBT algorithm
        choices = get_PBT_choices()
        #Randomly init model
        self.batch_size = int(np.random.choice(choices['batch_size']))
        self.model_type = np.random.choice(choices['model_type'])
        self.nf = np.random.choice(choices['nf'])
        self.hidden_layers = np.random.choice(choices['hidden_layers'])
        self.model = Model(self.size, self.output_nf, self.model_type, self.nf, self.hidden_layers)

        #Init optimiser
        lr = np.random.choice(choices['lr'])
        momentum = np.random.choice(choices['momentum'])
        self.optim = optim.Adam(self.model.parameters(), lr=lr, betas=(momentum,0.999))
    else:
        self.learning = True
        #Init model when PBT is not used
        self.model = Model(self.size, self.output_nf, self.model_type, self.nf, self.hidden_layers)
        self.optim = optim.Adam(self.model.parameters(), self.learning_rate, betas=(0.5, 0.999))


#Saves the entire state of the training process to continue afterwards or load best model
def save_state(self, file):
    path = os.path.join(self.result_root,file)
    torch.save({
            'iter': self.iter,
            'size': self.size,
            'model_type': self.model_type,
            'nf': self.nf,
            'hidden_layers': self.hidden_layers,
            'learning': self.learning,
            'batch_size': self.batch_size,
            'min_dist': self.min_dist,
            'best_iter': self.best_iter,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'TX_config': self.TX_config,
            'TX_input': self.TX_input,
            'blockage': self.blockage,
            'output_nf': self.output_nf
            }, path)
