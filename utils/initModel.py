import torch.nn as nn
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def initModel(self, device):
    resultpath = os.path.join(self.result_root,'checkpoint.pth')
    if os.path.isfile(resultpath):
        print("Restoring checkpoint")
        checkpoint = torch.load(resultpath, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
    else:
        self.model.apply(weights_init)

    self.model.to(device)

def saveCheckpoint(self):
    path = os.path.join(self.result_root,'checkpoint.pth')
    torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'loss': self.loss,
            }, path)
