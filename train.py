from utils.logger import setup_logger
from datasets.make_dataloader import make_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg_base as cfg

def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")    # 创建一个命令行参数解析器,用于支持 python train.py --config_file xxx.yml ...
    parser.add_argument(    # 指定 yaml 配置文件路径
        "--config_file", default="configs/person/vit_base.yml", help="path to config file", type=str
    )

    # 命令行覆盖配置（opts）,允许你在命令行中 临时覆盖 yaml 里的配置
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)   #argparse.REMAINDER 表示：剩下的所有参数都原样接收
    # 分布式训练的 rank,用于 多卡分布式训练（DDP）,每个进程对应一张 GPU,local_rank 表示当前进程使用第几张卡
    parser.add_argument("--local_rank", default=0, type=int)
    # 解析参数
    args = parser.parse_args()

    # 从 yaml 加载配置
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # 用命令行参数覆盖配置
    cfg.merge_from_list(args.opts)
    # 冻结配置,防止训练中途被误改
    cfg.freeze()

    device = torch.device(cfg.MODEL.DEVICE)

    # 设置随机种子
    set_seed(cfg.SOLVER.SEED, device)

    # 设置当前 GPU,在 DDP 下，每个进程绑定一张 GPU,防止多个进程抢同一张卡
    if cfg.MODEL.DIST_TRAIN and device.type == "cuda":
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建日志记录器
    logger = setup_logger("transreid", output_dir, if_train=True)
    # 打印关键信息
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    # 打印完整配置文件内容
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    # 打印最终生效配置
    logger.info("Running with config:\n{}".format(cfg))

    # 初始化进程通信,nccl：NVIDIA GPU 最快通信后端; env://：从环境变量中读取 rank / world_size
    if cfg.MODEL.DIST_TRAIN and device.type == "cuda":
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )

    # CUDA&数据加载
    if device.type == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 构建数据加载器
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 构建模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model = model.to(device)

    # 构建损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    if center_criterion is not None:
        center_criterion = center_criterion.to(device)
    # 构建优化器
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # 学习率调度器
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    # 启动训练主循环
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
