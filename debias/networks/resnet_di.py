import torch.nn as nn

from torchvision.models import resnet18, resnet50


class DIResNet18(nn.Module):
    def __init__(self, num_classes=2, num_biases=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes * num_biases)
        print(f'DIResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        return logits


class DIResNet50(nn.Module):
    def __init__(self, num_classes=2, num_biases=2, pretrained=True,hidden_size=2048, dropout=0.5):
        super().__init__()
        
        self.resnet = resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes * num_biases)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        
        print(f'DIResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs
    