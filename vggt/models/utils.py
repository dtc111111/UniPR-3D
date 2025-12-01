import torch
import torch.nn as nn

def freeze_module(module):
    module.eval()
    if isinstance(module, nn.Parameter):
        module.requires_grad = False
        return module
    
    for param in module.parameters():
        param.requires_grad = False
    return module

def unfreeze_module(module):
    module.train()
    if isinstance(module, nn.Parameter):
        module.requires_grad = True
        return module
    
    for param in module.parameters():
        param.requires_grad = True
    return module