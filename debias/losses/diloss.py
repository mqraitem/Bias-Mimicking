import torch
import torch.nn as nn
import torch.nn.functional as F


class DILoss(nn.Module):
    def __init__(self, num_classes=2, num_biases=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_biases = num_biases
        print(f'DILoss')

    def forward(self, logits, labels, biases, gc=None):
        logprobs = [F.log_softmax(logits[:, i * self.num_classes: (i + 1) * self.num_classes], dim=1) for i in
                    range(self.num_biases)]
        output = torch.cat(logprobs, dim=1)
        target = biases * self.num_classes + labels
        if gc is None: 
            return F.nll_loss(output, target)
        else:   
            loss = F.nll_loss(output, target, reduction='none') 
            loss *= gc 
            loss = torch.mean(loss) 
            return loss

class DILossBinary(nn.Module):

    def __init__(self, num_classes=2, num_biases=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_biases = num_biases
        print(f'DILoss')

    def forward(self, logits, labels, biases):

        domain_label = biases[:, None]
        class_num = logits.size(1) // 2
        loss = F.binary_cross_entropy_with_logits(
                    domain_label*logits[:, :class_num]
                        + (1-domain_label)*logits[:, class_num:],
                    labels)
        return loss
