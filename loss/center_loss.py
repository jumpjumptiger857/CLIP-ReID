from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """
    Center loss.
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=751, feat_dim=2048):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # ⭐ centers 会自动跟随 .to(device)
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim)
        )

    def forward(self, x, labels):
        """
        Args:
            x: (batch_size, feat_dim)
            labels: (batch_size,)
        """
        assert x.size(0) == labels.size(0), \
            "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)

        # ⭐ 保证 classes 和 x 在同一个 device
        device = x.device
        classes = torch.arange(self.num_classes, device=device).long()

        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)
            dist.append(value)

        dist = torch.cat(dist)
        loss = dist.mean()
        return loss
