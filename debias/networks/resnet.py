import torch.nn as nn
import torch.nn.functional as F
import torch 

from torchvision.models import resnet18, resnet50


class FCResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)
        print(f'FCResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)

        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        return logits, feat

class FCResNet18Base(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        print(f'FCResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        feat = F.normalize(out, dim=1)
        return feat

class FCResNet50(nn.Module):    
    def __init__(self, num_classes=2, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs, features

class FCResNet18_Base(nn.Module):   
    """ResNet50 but without the final fc layer"""
    
    def __init__(self, pretrained=True, hidden_size=512, dropout=0.5):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(512, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(self.relu(features))

        return features