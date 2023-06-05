import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from DenseNet import *

class ResNet18(nn.Module):
    def __init__(self, K, loss_type, input_ch):
        # Following same model setup of Papyan et al [2020]
        super(ResNet18, self).__init__()
        self.loss_type = loss_type
        self.core_model = models.resnet18(pretrained=False, num_classes=K)
        self.core_model.conv1 = nn.Conv2d(input_ch, self.core_model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
        self.core_model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        
        self.core_model.fc = nn.Identity()
        self.normalizer = F.normalize
    
    def forward(self, x):
        x = self.core_model(x)
        x = self.normalizer(x, dim = 1)
        return x


class DenseNet40(nn.Module):
    def __init__(self, K, loss_type, input_ch, old_deepnet = False):
        # Following same model setup of Papyan et al [2020]
        super(DenseNet40, self).__init__()
        self.loss_type = loss_type
        
        if input_ch == 1:
          self.core_model = DenseNet40_Base(num_classes=10, grayscale=True, growth_rate = 12, block_config = (6,6,6))
        else:
          self.core_model = DenseNet40_Base(num_classes=10, grayscale=False, growth_rate = 12, block_config = (6,6,6))
        
        self.core_model.classifier = nn.Identity()
        self.normalizer = F.normalize
    
    def forward(self, x):
        x = self.core_model(x)
        x = self.normalizer(x, dim = 1)
        return x