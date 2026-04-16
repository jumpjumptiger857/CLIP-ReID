import logging
import os
import time
import torch
from torch.cuda import amp
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist

# 配置校验函数
def validate_config(cfg):
    assert cfg.SOLVER.MAX_EPOCHS > 0, "MAX_EPOCHS must be > 0"
    assert cfg.SOLVER.LOG_PERIOD > 0, "LOG_PERIOD must be > 0"
    assert cfg.SOLVER.CHECKPOINT_PERIOD >= 0, "CHECKPOINT_PERIOD must >= 0"
    assert cfg.SOLVER.EVAL_PERIOD >= 0, "EVAL_PERIOD must >= 0"
    # 若配置为0，给出警告并设置默认值
    if cfg.SOLVER.CHECKPOINT_PERIOD == 0:
        logging.warning("CHECKPOINT_PERIOD is 0, set to default 10")
        cfg.SOLVER.CHECKPOINT_PERIOD = 10
    if cfg.SOLVER.EVAL_PERIOD == 0:
        logging.warning("EVAL_PERIOD is 0, set to default 5")
        cfg.SOLVER.EVAL_PERIOD = 5

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    # 读取配置参数（日志/保存/验证频率）
    log_period = cfg.SOLVER.LOG_PERIOD  # 日志打印频率（如每100步打印一次）
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  # 模型保存频率（如每10个epoch保存一次）
    eval_period = cfg.SOLVER.EVAL_PERIOD  # 验证频率（如每5个epoch验证一次）

    device = torch.device(cfg.MODEL.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCHS

    #  日志初始化
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    # model.to(device) =
    # “告诉 PyTorch：这个模型接下来在哪个设备上干活”
    model.to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1 and not cfg.MODEL.DIST_TRAIN:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    #  指标记录器（滑动平均loss/acc）
    loss_meter = AverageMeter()  # 记录loss，自动计算平均值
    acc_meter = AverageMeter()  # 记录训练准确率

    # R1_mAP_eval：ReID 专用评估（CMC + mAP）
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # AMP（混合精度）
    use_amp = device.type == "cuda"
    if use_amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()  # 每个epoch重置loss记录
        acc_meter.reset()  # 每个epoch重置acc记录
        evaluator.reset()  # 每个epoch重置评估器



        #  模型设为训练模式（启用Dropout/BatchNorm训练行为）
        model.train()

        #  遍历训练集loader
        # img：图像Tensor [B, C, H, W]，vid：person id（行人 ID），target_cam：摄像头 ID，target_view：视角 ID
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
            # 数据搬到 GPU + 条件控制
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            #  前向传播 + 反向传播（分AMP/非AMP）
            if use_amp:
                # 前向 + loss（AMP 版本）
                with amp.autocast(enabled=use_amp):
                    score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    loss = loss_fn(score, feat, target, target_cam)
                # 反向传播（AMP）
                scaler.scale(loss).backward()
                scaler.step(optimizer)      # 更新主模型参数
                if optimizer_center is not None:
                    scaler.step(optimizer_center)
                scaler.update()             # 更新缩放器状态
            else:
                # 普通精度训练
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)
                loss.backward()
                optimizer.step()

            # Center Loss 梯度修正
            if optimizer_center is not None and center_criterion is not None:
                # CenterLoss的梯度需要除以权重系数，避免梯度过大
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)

            # scheduler.step()：按预先设定的策略，更新当前 optimizer 的学习率
            scheduler.step()

            # 训练准确率  （只是一个辅助指标）
            #  这个指标只是用来看训练有没有在正常跑，
            #  不是模型真正好坏的评判标准，
            #  不参与反向传播，也不决定最终性能。
            if isinstance(score, list):
                # 多分支输出时，取第一个分支计算acc
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            # 更新指标记录器
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            #  GPU同步（保证时间统计准确）
            if device.type == "cuda":
                torch.cuda.synchronize()

            #  打印日志（按log_period频率
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        #  统计当前epoch的训练速度
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # 1. 模型保存（按checkpoint_period频率）
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 分布式训练：仅主进程（rank=0）保存
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                # 单机训练：直接保存
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # 训练前调用校验
        validate_config(cfg)

        # 2. 验证评估（按eval_period频率）
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 分布式训练：仅主进程验证
                if dist.get_rank() == 0:
                    model.eval()  # 模型设为评估模式（禁用Dropout/BatchNorm训练行为）
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():  # 禁用梯度计算，节省显存
                            img = img.to(device)
                            # 处理摄像头/视角ID
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            # 前向传播：仅提取特征（无分类score）
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))  # 记录特征/标签/摄像头ID
                    # 计算ReID核心指标
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    # 打印验证结果
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    # 清空GPU缓存
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            else:
                # 单机训练：验证流程
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = torch.device(cfg.MODEL.DEVICE)
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    model.to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]