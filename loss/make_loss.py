import torch
import torch.nn.functional as F
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss(cfg, num_classes):
    """
    CPU minimal loss:
    only softmax cross entropy
    """

    # score	分类输出（logits）； feat	特征向量（本来给 triplet 用）； target	行人 ID（标签）； target_cam	摄像头 ID（这里没用）
    def loss_func(score, feat, target, target_cam=None):
        """
        score: [cls_score, cls_score_proj] or Tensor
        """
        if isinstance(score, list):
            score = score[0]

        return F.cross_entropy(score, target)

    center_criterion = None
    # loss_func：主 loss； center_criterion：Center Loss（单独优化）
    return loss_func, center_criterion
