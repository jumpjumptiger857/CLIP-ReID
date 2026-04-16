import torch

#  make_optimizer 根据配置，把模型中“需要训练的参数”分组，
# 为每一组设置合适的学习率和 weight decay，
# 并返回主模型 optimizer +（可选）center loss optimizer。
def make_optimizer(cfg, model, center_criterion=None):
    # 手动构建params列表，因为ReID / Transformer 训练中，不同参数需要不同优化策略
    params = []

    for key, value in model.named_parameters():
        # 跳过不需要训练的参数
        if not value.requires_grad:
            continue

        # 默认学习率 & 权重衰减
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            momentum=cfg.SOLVER.MOMENTUM
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(
            params
        )
    else:
        raise ValueError("Unsupported optimizer: {}".format(cfg.SOLVER.OPTIMIZER_NAME))

    # ===============================
    # center optimizer（关键修复点）
    # Center Loss 要单独一个 optimizer原因：
    # Center Loss 的“中心向量”不是模型参数的一部分，它不在model.parameters(),是一个独立的learnable tensor,所以不能交给主optimizer，必须单独优化
    # ===============================
    optimizer_center = None
    if center_criterion is not None:
        optimizer_center = torch.optim.SGD(
            center_criterion.parameters(),
            lr=cfg.SOLVER.CENTER_LR
        )

    # optimizer：更新模型参数； optimizer_center：更新中心向量（可选）
    return optimizer, optimizer_center
