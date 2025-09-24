
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
from config.choices import sample_choices, network_choices, optim_choices, act_choices, lr_func_choices, \
    image_format_choices, noise_schedule_choices, parse_image_size_type, loss_func_choices, autoencoder_network_choices
from model.trainers import DMTrainer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def main(args):

    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(DMTrainer(args=args).train, nprocs=gpus)
    else:
        DMTrainer(args=args).train()


def init_train_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditional", "-c", default=True, action="store_true")
    parser.add_argument("--latent", "-l", default=True, action="store_true")
    parser.add_argument("--sample", "-s", type=str, default="ddpm", choices=sample_choices)
    parser.add_argument("--network", type=str, default="unet", choices=network_choices)
    parser.add_argument("--run_name", "-n", type=str, default="df")
    parser.add_argument("--epochs", "-e", type=int, default=300)
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", "-i", type=parse_image_size_type, default=64)
    parser.add_argument("--dataset_path", type=str, default="D:\\code\\Diffusion-Models-pytorch-main\\Image\\0")
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--optim", type=str, default="adamw", choices=optim_choices)
    parser.add_argument("--loss", type=str, default="mse", choices=loss_func_choices)
    parser.add_argument("--act", type=str, default="gelu", choices=act_choices)
    parser.add_argument("--lr_func", type=str, default="linear", choices=lr_func_choices)
    parser.add_argument("--result_path", type=str, default="D:\\code\\Integrated-Design-Diffusion-Model-main\\results_LDM-5")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_model_interval", default=True, action="store_true")
    parser.add_argument("--save_model_interval_epochs", type=int, default=10)
    parser.add_argument("--start_model_interval", type=int, default=1)
    parser.add_argument("--vis", "-v", default=True, action="store_true")
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=noise_schedule_choices)
    parser.add_argument("--resume", "-r", default=False, action="store_true")
    parser.add_argument("--start_epoch", type=int, default=None)
    parser.add_argument("--pretrain", default=False, action="store_true")
    parser.add_argument("--pretrain_path", type=str, default="")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--main_gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--cfg_scale", type=int, default=3)
    parser.add_argument("--autoencoder_network", type=str, default="vae", choices=autoencoder_network_choices)
    parser.add_argument("--autoencoder_ckpt", type=str, 
        default="D:\\code\\Integrated-Design-Diffusion-Model-main\\results\\autoencoder\\ckpt_last.pt")

    return parser.parse_args()


if __name__ == "__main__":
    args = init_train_args()
    main(args)
