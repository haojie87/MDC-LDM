import sys
import os
root_file = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(root_file)

import sys
import logging
import time

import coloredlogs
import torch

from torch import nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from config.setting import MASTER_ADDR, MASTER_PORT, LATENT_CHANNEL
from model.trainers.base import Trainer
from utils.check import check_is_distributed
from utils.checkpoint import load_ckpt, save_ckpt
from utils.initializer import seed_initializer, device_initializer, optimizer_initializer, amp_initializer, \
    loss_initializer, lr_initializer, autoencoder_network_initializer
from utils.utils import setup_logging, save_train_logging, check_and_create_dir, save_images
from utils.datasetvae import get_dataset, post_image

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class AutoencoderTrainer(Trainer):
    """
    自动编码器训练器
    """

    def __init__(self, **kwargs):
        super(AutoencoderTrainer, self).__init__(**kwargs)
        # 自动编码器参数
        self.run_name = self.check_args_and_kwargs(kwarg="run_name", default="autoencoder")
        # 数据集
        self.train_dataset_path = self.check_args_and_kwargs(kwarg="train_dataset_path", default="")
        self.val_dataset_path = self.check_args_and_kwargs(kwarg="val_dataset_path", default="")
        # 潜在空间参数
        self.latent_channels = LATENT_CHANNEL

        # 默认参数
        self.train_dataloader = None
        self.val_dataloader = None
        self.len_train_dataloader = None
        self.len_val_dataloader = None
        self.save_val_vis_dir = None
        self.best_score = 0
        self.avg_train_loss = 0
        self.avg_val_loss = 0
        self.avg_score = 0

    def before_train(self):
        """
        训练自动编码器模型前的方法
        """
        logger.info(msg=f"[{self.rank}]: 开始自动编码器模型训练")
        logger.info(msg=f"[{self.rank}]: 输入参数: {self.args}")
        # 第一步：设置路径并创建日志
        # 创建数据日志路径
        self.results_logging = setup_logging(save_path=self.result_path, run_name=self.run_name)
        self.results_dir = self.results_logging[1]
        self.results_vis_dir = self.results_logging[2]
        self.results_tb_dir = self.results_logging[3]
        self.args = save_train_logging(arg=self.args, save_path=self.results_dir)

        # 第二步：获取初始化器和参数的参数
        # 初始化种子
        seed_initializer(seed_id=self.seed)
        # 初始化并保存模型标识位
        # 在这里检查是单GPU训练还是多GPU训练
        self.save_models = True
        # 是否启用分布式训练
        if check_is_distributed(distributed=self.distributed):
            self.distributed = True
            # 设置地址和端口
            os.environ["MASTER_ADDR"] = MASTER_ADDR
            os.environ["MASTER_PORT"] = MASTER_PORT
            # 进程总数等于显卡数量
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=self.rank,
                                    world_size=self.world_size)
            # 设置设备ID
            self.device = device_initializer(device_id=self.rank, is_train=True)
            # 可能有随机错误，使用此函数可以减少cudnn中的随机错误
            # torch.backends.cudnn.deterministic = True
            # 分布式训练期间的同步
            dist.barrier()
            # 如果分布式训练不是主GPU，则保存模型标志为False
            if dist.get_rank() != self.main_gpu:
                self.save_models = False
            logger.info(msg=f"[{self.device}]: 成功使用分布式训练。")
        else:
            self.distributed = False
            # 运行设备初始化器
            self.device = device_initializer(device_id=self.use_gpu, is_train=True)
            logger.info(msg=f"[{self.device}]: 成功使用普通训练。")

        # 第三步：设置数据
        self.train_dataloader = get_dataset(image_size=self.image_size, dataset_path=self.train_dataset_path,
                                            batch_size=self.batch_size, num_workers=self.num_workers,
                                            distributed=self.distributed)
        self.val_dataloader = get_dataset(image_size=self.image_size, dataset_path=self.val_dataset_path,
                                          batch_size=self.batch_size, num_workers=self.num_workers,
                                          distributed=self.distributed)
                                          
        # 第四步：初始化模型
        Network = autoencoder_network_initializer(network=self.network, device=self.device)
        self.model = Network(latent_channels=self.latent_channels, device=self.device).to(self.device)
        # 分布式训练
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device],
                                                             find_unused_parameters=True)
        # 模型优化器
        self.optimizer = optimizer_initializer(model=self.model, optim=self.optim, init_lr=self.init_lr,
                                               device=self.device)
        if self.resume:
            ckpt_path = None
            # 确定加载哪个检查点
            # 'start_epoch'是正确的
            if self.start_epoch is not None:
                ckpt_path = os.path.join(self.results_dir, f"ckpt_{str(self.start_epoch - 1).zfill(3)}.pt")
            # 在训练模式下，参数'ckpt_path'为None
            if ckpt_path is None:
                ckpt_path = os.path.join(self.results_dir, "ckpt_last.pt")
            # 获取模型状态
            self.start_epoch, model_score = load_ckpt(ckpt_path=ckpt_path, model=self.model, device=self.device,
                                                      optimizer=self.optimizer, is_distributed=self.distributed,
                                                      ckpt_type="autoencoder")
            self.best_score = model_score[0]
            logger.info(msg=f"[{self.device}]: 成功加载恢复模型检查点。")
        else:
            # 预训练模式
            if self.pretrain:
                load_ckpt(ckpt_path=self.pretrain_path, model=self.model, device=self.device, is_pretrain=self.pretrain,
                          is_distributed=self.distributed)
                logger.info(msg=f"[{self.device}]: 成功加载预训练模型检查点。")
            self.start_epoch, self.best_score = 0, 0
        logger.info(msg=f"[{self.device}]: 起始epoch为{self.start_epoch}，最佳分数为{self.best_score}。")

        # 设置半精度
        self.scaler = amp_initializer(amp=self.amp, device=self.device)
        # 损失函数
        self.loss_func = loss_initializer(loss_name=self.loss_name, device=self.device)
        # Tensorboard
        self.tb_logger = SummaryWriter(log_dir=self.results_tb_dir)
        # 数据加载器中数据集批次的数量
        self.len_train_dataloader = len(self.train_dataloader)
        self.len_val_dataloader = len(self.val_dataloader)

    def before_iter(self):
        """
        训练自动编码器模型一个迭代前的方法
        """
        logger.info(msg=f"[{self.device}]: 开始epoch {self.epoch}:")
        # 设置学习率
        current_lr = lr_initializer(lr_func=self.lr_func, optimizer=self.optimizer, epoch=self.epoch,
                                    epochs=self.epochs,
                                    init_lr=self.init_lr, device=self.device)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: 当前学习率", scalar_value=current_lr, global_step=self.epoch)
        # 创建可视化目录
        self.save_val_vis_dir = os.path.join(self.results_vis_dir, str(self.epoch))
        check_and_create_dir(self.save_val_vis_dir)

    def train_in_iter(self):
        """
        训练自动编码器模型一个迭代中的方法
        """
        # 初始化图像和标签
        train_loss_list, val_loss_list, score_list = [], [], []
        # 训练
        self.model.train()
        logger.info(msg="开始训练模式。")
        train_pbar = tqdm(self.train_dataloader)
        for i, (images, _) in enumerate(train_pbar):
            # 输入图像 [B, C, H, W]
            images = images.to(self.device)
            with autocast(enabled=self.amp):
                recon_images = self.model(images)
                # 计算MSE损失
                train_loss = self.loss_func(recon_images, images)
            # 优化器清除模型参数的梯度
            self.optimizer.zero_grad()
            # 更新损失和优化器
            # Fp16 + Fp32
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # TensorBoard日志记录
            train_pbar.set_postfix(MSE=train_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: 训练损失({self.loss_func})",
                                      scalar_value=train_loss.item(),
                                      global_step=self.epoch * self.len_train_dataloader + i)
            train_loss_list.append(train_loss.item())
            # 每个epoch的损失
        self.avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: 训练损失",
                                  scalar_value=self.avg_train_loss,
                                  global_step=self.epoch)
        logger.info(f"训练损失: {self.avg_train_loss}")
        logger.info(msg="完成训练模式。")

        # 验证
        self.model.eval()
        logger.info(msg="开始验证模式。")
        val_pbar = tqdm(self.val_dataloader)
        for i, (images, _) in enumerate(val_pbar):
            # 输入图像 [B, C, H, W]
            images = images.to(self.device)
            with autocast(enabled=self.amp):
                recon_images = self.model(images)
                # 计算MSE损失
                val_loss = self.loss_func(recon_images, images)
            # 优化器清除模型参数的梯度
            self.optimizer.zero_grad()
            # TensorBoard日志记录
            val_pbar.set_postfix(MSE=val_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: 验证损失({self.loss_func})", scalar_value=val_loss.item(),
                                      global_step=self.epoch * self.len_val_dataloader + i)
            val_loss_list.append(val_loss.item())
            # TODO: 后期增加评估指标
            score = 0
            self.tb_logger.add_scalar(tag=f"[{self.device}]: 分数({self.loss_func})", scalar_value=score,
                                      global_step=self.epoch * self.len_val_dataloader + i)
            score_list.append(score)
            images = post_image(images=images, device=self.device)
            if self.loss_name == "mse_kl":
                recon_images = recon_images[0]
            recon_images = post_image(images=recon_images, device=self.device)
            image_name = time.time()
            for index, image in enumerate(images):
                save_images(images=image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{index}_origin.{self.image_format}"))
            for recon_index, recon_image in enumerate(recon_images):
                save_images(images=recon_image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{recon_index}_recon.{self.image_format}"))
        # 每个epoch的损失和分数
        self.avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        self.avg_score = sum(score_list) / len(score_list)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: 验证损失", scalar_value=self.avg_val_loss,
                                  global_step=self.epoch)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: 平均分数", scalar_value=self.avg_score,
                                  global_step=self.epoch)
        logger.info(f"验证损失: {self.avg_val_loss}, 分数: {self.avg_score}")
        logger.info(msg="完成验证模式。")

    def after_iter(self):
        """
        训练自动编码器模型一个迭代后的方法
        """
        # 在主进程中保存和验证模型
        if self.save_models:
            # 保存模型，设置检查点名称
            save_name = f"ckpt_{str(self.epoch).zfill(3)}"
            # 初始化检查点参数
            ckpt_model = self.model.state_dict()
            ckpt_optimizer = self.optimizer.state_dict()
            # 保存最佳模型
            if self.avg_score > self.best_score:
                is_best = True
                self.best_score = self.avg_score
            else:
                is_best = False
            save_ckpt(epoch=self.epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=None,
                      ckpt_optimizer=ckpt_optimizer, results_dir=self.results_dir,
                      save_model_interval=self.save_model_interval,
                      save_model_interval_epochs=self.save_model_interval_epochs,
                      start_model_interval=self.start_model_interval, image_size=self.image_size,
                      network=self.network, act=self.act, is_autoencoder=True, is_best=is_best, score=self.avg_score,
                      best_score=self.best_score, latent_channel=self.latent_channels)
            logger.info(msg=f"[{self.device}]: 完成epoch {self.epoch}:")
        # 分布式训练期间的同步
        if self.distributed:
            logger.info(msg=f"[{self.device}]: 分布式训练期间的同步。")
            dist.barrier()

    def after_train(self):
        """
        训练自动编码器模型后的方法
        """
        logger.info(msg=f"[{self.device}]: 完成训练。")
        # 清理分布式环境
        if self.distributed:
            dist.destroy_process_group()
