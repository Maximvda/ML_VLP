import torch
import torch.nn as nn
import os

from models.architecture import model

#Initialises a model from the cnn architecture for a given input size
def initModel(data_loader, model_type, nf, extra_layers):
    input, output = next(iter(data_loader))
    return model(input.size(2), model_type, nf, extra_layers)

#Initialises the weights of the model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Saves the best performing model to model.pth file
def saveBestModel(result_root, model):
    path = os.path.join(result_root,'model.pth')
    torch.save({'model': model.state_dict()}, path)

#Loads the best model from the saved path and loads onto device
def loadBestModel(result_root, model, device):
    path = os.path.join(result_root,'model.pth')
    if os.path.isfile(path):
        print("Restoring best model")
        save = torch.load(path, map_location=device)
        model.load_state_dict(save['model'])

#Loads the entire state of the training process to continue afterwards
def loadCheckpoint(self, device):
    resultpath = os.path.join(self.result_root,'checkpoint.pth')
    if os.path.isfile(resultpath):
        print("Restoring checkpoint")
        checkpoint = torch.load(resultpath, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.learning = checkpoint['learning']
        self.min_distance = checkpoint['min_distance']
        self.distance = checkpoint['distance']
        self.best_epoch = checkpoint['best_epoch']
        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        self.model.apply(weights_init)

    self.model.to(device)

#Saves the entire state of the training process to continue afterwards
def saveCheckpoint(self):
    path = os.path.join(self.result_root,'checkpoint.pth')
    torch.save({
            'epoch': self.epoch,
            'learning': self.learning,
            'distance': self.distance,
            'min_distance': self.min_distance,
            'best_epoch': self.best_epoch,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'loss': self.loss,
            }, path)
