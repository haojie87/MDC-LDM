
import os
import sys
import argparse
import logging

import coloredlogs
import torch

from torch import multiprocessing as mp

sys.path.append(os.path.dirname(sys.path[0]))
root_file = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(root_file)
from config.choices import autoencoder_network_choices, optim_choices, autoencoder_loss_func_choices, \
    image_format_choices

from model.trainers.trainer_VAE import AutoencoderTrainer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def main(args):
    
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(AutoencoderTrainer(args=args).train, nprocs=gpus)
    else:
        AutoencoderTrainer(args=args).train()


def init_autoencoder_train_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--network", type=str, default="vae", choices=autoencoder_network_choices)
    parser.add_argument("--run_name", "-n", type=str, default="autoencoder")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", "-i", type=int, default=512)
    parser.add_argument("--latent_channels", "-l", type=int, default=8)
    parser.add_argument("--train_dataset_path", type=str,
                        default="D:\\code\\Diffusion-Models-pytorch-main\\Image")
    parser.add_argument("--val_dataset_path", type=str,
                        default="D:\\code\\Diffusion-Models-pytorch-main\\Image_val")
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--optim", type=str, default="adamw", choices=optim_choices)
    parser.add_argument("--loss", type=str, default="mse_kl", choices=autoencoder_loss_func_choices)
    parser.add_argument("--act", type=str, default="silu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_func", type=str, default="cosine")
    parser.add_argument("--result_path", type=str, default="D:\\code\\Integrated-Design-Diffusion-Model-main\\results")
    parser.add_argument("--save_model_interval", default=False, action="store_true")
    parser.add_argument("--save_model_interval_epochs", type=int, default=10)
    parser.add_argument("--start_model_interval", type=int, default=10)
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    parser.add_argument("--resume", "-r", default=False, action="store_true")
    parser.add_argument("--start_epoch", type=int, default=None)
    parser.add_argument("--pretrain", default=False, action="store_true")
    parser.add_argument("--pretrain_path", type=str, default="")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--distributed", "-d", default=False, action="store_true")
    parser.add_argument("--main_gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_autoencoder_train_args()
    main(args)
