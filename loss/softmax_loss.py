import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes)
        targets: (batch_size,)
        """
        device = inputs.device

        log_probs = self.logsoftmax(inputs)

        # one-hot, follow inputs device
        targets_onehot = torch.zeros_like(log_probs, device=device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

        targets_onehot = (1 - self.epsilon) * targets_onehot + \
                          self.epsilon / self.num_classes

        loss = (-targets_onehot * log_probs).mean(0).sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()