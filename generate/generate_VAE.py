
import os
import sys
import argparse
import time

import torch
import logging
import coloredlogs

sys.path.append(os.path.dirname(sys.path[0]))
root_file = os.path.dirname(sys.path[0])
sys.path.append(root_file)
from config import IMAGE_CHANNEL
from config.choices import sample_choices, image_format_choices, parse_image_size_type
from utils.check import check_image_size
from utils.initializer import device_initializer, network_initializer, sample_initializer, generate_initializer, \
    generate_autoencoder_initializer, autoencoder_network_initializer
from utils.utils import plot_images, save_images, save_one_image_in_images, check_and_create_dir, save_one_image_in_images_paper
from utils.checkpoint import load_ckpt

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

from iddm.config.choices import network_choices, act_choices


import cv2
from scipy.spatial import distance
import numpy as np


def sample_points(image: np.ndarray, num_points: int):
    """
    在图像中随机不重复采点
    """
    H, W = image.shape[:2]
    total_pixels = H * W
    assert num_points <= total_pixels, "采样点数量不能超过像素总数"
    # 随机选择像素索引（不重复）
    flat_indices = np.random.choice(total_pixels, size=num_points, replace=False)
    # 转为 (y, x) 坐标
    ys, xs = np.unravel_index(flat_indices, (H, W))
    coords = np.stack([ys, xs], axis=1)
    # 对应像素值
    values = image[ys, xs]
    return coords, values

def average_distance(coords):
    """
    # 计算采样点平均距离
    """
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    triu_idx = np.triu_indices(len(coords), k=1)
    return dist_matrix[triu_idx].mean()


def Points_Radiation(image, image_size=512, num_points=100, dis_ratio=0.2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_org, w_org = image.shape[:2]
    #  采样
    coords, values = sample_points(image, num_points)
    # 3. resize
    image_resize = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    H, W = image_resize.shape[:2]
    zoom_h = image_size // h_org
    zoom_w = image_size // h_org
    coords_scaled = coords * np.array([zoom_h, zoom_w]) # 计算放大后对应的坐标
    values_scaled = values / 255
    avg_dist = average_distance(coords_scaled) # 平均井点距离
    radius = dis_ratio * avg_dist # 辐射范围
    img_points_radiation = np.zeros((H, W), dtype=np.float32) 
    # 对每个采样点进行辐射赋值
    for (cy, cx), val in zip(coords_scaled, values_scaled):
        y_min = max(0, int(cy - radius))
        y_max = min(H, int(cy + radius) + 1)
        x_min = max(0, int(cx - radius))
        x_max = min(W, int(cx + radius) + 1)
        # 局部坐标网格
        yy, xx = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
        dist_to_center = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        # 计算置信度（线性衰减）
        confidence = np.clip(1 - dist_to_center / radius, 0, 1)
        # 辐射值 = 中心值 × 置信度
        new_values = val * confidence
        # 累加模式（可以改成取最大值）
        img_points_radiation[y_min:y_max, x_min:x_max] = np.maximum(img_points_radiation[y_min:y_max, x_min:x_max], new_values)
    return img_points_radiation, coords_scaled, values


class Generator:
    """
    Diffusion model generator
    """

    def __init__(self, gen_args, deploy=False):
        """
        Generating init
        :param gen_args: Input parameters
        :param deploy: App deploy
        :return: None
        """
        self.args = gen_args
        self.deploy = deploy

        logger.info(msg="Start generation.")
        logger.info(msg=f"Input params: {self.args}")
        # Init parameters
        self.in_channels, self.out_channels = IMAGE_CHANNEL, IMAGE_CHANNEL
        (self.autoencoder, self.autoencoder_network, self.autoencoder_image_size, self.autoencoder_latent_channels,
         self.autoencoder_act) = None, None, None, None, None

        # Latent diffusion model
        self.latent = self.args.latent
        # Weight path
        self.weight_path = self.args.weight_path
        # Autoencoder weight path
        self.autoencoder_ckpt = self.args.autoencoder_ckpt
        # Run device initializer
        self.device = device_initializer(device_id=self.args.use_gpu)

        # Enable conditional generation, sample type, network, image size,
        # number of classes and select activation function
        gen_results = generate_initializer(ckpt_path=self.weight_path, conditional=self.args.conditional,
                                           image_size=self.args.image_size, sample=self.args.sample,
                                           network=self.args.network, act=self.args.act,
                                           num_classes=self.args.num_classes, device=self.device)
        self.conditional, self.network, self.image_size, self.num_classes, self.act = gen_results


        # Check image size format
        self.image_size = check_image_size(image_size=self.image_size)
        self.resize_image_size = self.image_size
        # Generation name
        self.generate_name = self.args.generate_name
        # Sample
        self.sample = self.args.sample
        # Number of images
        self.num_images = self.args.num_images
        # Use ema
        self.use_ema = self.args.use_ema
        # Format of images
        self.image_format = self.args.image_format
        # Saving path
        self.result_path = os.path.join(self.args.result_path, str(time.time()))
        # Check and create result path
        if not deploy:
            check_and_create_dir(self.result_path)
        # Network
        self.Network = network_initializer(network=self.network, device=self.device)

        # If latent diffusion model is enabled, get autoencoder results
        if self.latent:
            gen_autoencoder_results = generate_autoencoder_initializer(ckpt_path=self.autoencoder_ckpt,
                                                                       device=self.device)
            (self.autoencoder_network, self.autoencoder_image_size, self.autoencoder_latent_channels,
             self.autoencoder_act) = gen_autoencoder_results
            self.in_channels, self.out_channels = self.autoencoder_latent_channels, self.autoencoder_latent_channels
            self.resize_image_size = check_image_size(image_size=self.autoencoder_image_size)

            # Init autoencoder network
            autoencoder_network = autoencoder_network_initializer(network=self.autoencoder_network, device=self.device)
            self.autoencoder = autoencoder_network(latent_channels=self.autoencoder_latent_channels,
                                                   device=self.device).to(self.device)
            load_ckpt(ckpt_path=self.autoencoder_ckpt, model=self.autoencoder, is_train=False, device=self.device)
            # Inference mode, no updating parameters
            self.autoencoder.eval()

        # Initialize the diffusion model
        self.diffusion = sample_initializer(sample=self.sample, image_size=self.image_size, device=self.device,
                                            latent=self.latent, latent_channel=self.autoencoder_latent_channels,
                                            autoencoder=self.autoencoder)

        # Is it necessary to expand the image?
        self.resize_image_size = check_image_size(image_size=self.resize_image_size)
        if self.image_size == self.resize_image_size:
            self.new_image_size = None
        else:
            self.new_image_size = self.resize_image_size
        # Initialize model
        if self.conditional:
            # Generation class name
            self.class_name = self.args.class_name
            # classifier-free guidance interpolation weight
            self.cfg_scale = self.args.cfg_scale
            # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
            # self.model = self.Network(in_channel=self.in_channels, out_channel=self.out_channels,
            #                           num_classes=self.num_classes, device=self.device, image_size=self.image_size,
            #                           act=self.act).to(self.device)
            self.model = self.Network(in_channel=self.in_channels, out_channel=self.out_channels,
                                 num_classes=self.num_classes, device=self.device, image_size=self.image_size,
                                 act=self.act).to(self.device)
            load_ckpt(ckpt_path=self.weight_path, model=self.model, device=self.device, is_train=False,
                      is_use_ema=self.use_ema, conditional=self.conditional)
        else:
            # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
            self.model = self.Network(device=self.device, image_size=self.image_size, act=self.act).to(self.device)
            load_ckpt(ckpt_path=self.weight_path, model=self.model, device=self.device, is_train=False,
                      conditional=self.conditional)

    def generate(self, path, num_points=100, index=0):
        """
        Generate images
        :param index: Image index
        """
        # img = cv2.imread("D:\\code\\Diffusion-Models-pytorch-main\\Image\\0\\2_2_460_Facies.png")
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        img_points_radiation, coords, values = Points_Radiation(img, image_size=64, num_points=num_points, dis_ratio=0.1)
        zoom_h = 512 // 64
        zoom_w = 512 // 64
        coords_scaled = coords * np.array([zoom_h, zoom_w]) # 计算放大后对应的坐标
        img_points_radiation = torch.from_numpy(img_points_radiation).float()
        img_points_radiation = img_points_radiation.unsqueeze(0)
        img_points_radiation = img_points_radiation.unsqueeze(0)
        labels = img_points_radiation.repeat(1, 8, 1, 1).to(self.device)
        x = self.diffusion.sample(model=self.model, n=self.num_images, labels=labels, cfg_scale=self.cfg_scale)

        # If deploy app is true, return the generate results
        if self.deploy:
            return x

        if self.result_path == "" or self.result_path is None:
            plot_images(images=x)
        else:
            save_name = f"{self.generate_name}_{index}"
            # save_images(images=x, path=os.path.join(self.result_path, f"{save_name}.{self.image_format}"))
            # save_one_image_in_images(images=x, path=self.result_path, generate_name=save_name,
            #                          image_size=self.new_image_size, image_format=self.image_format)
            # plot_images(images=x)

            save_one_image_in_images_paper(images=x, path=self.result_path, generate_name=save_name,
                                     image_size=self.new_image_size, image_format=self.image_format)
            # 保存到一个 npz 文件中
            np.savez(os.path.join(self.result_path, save_name+".npz"), array1=coords_scaled, array2=values)
        logger.info(msg="Finish generation.")


def init_generate_args():
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Generation name (required)
    parser.add_argument("--generate_name", "-n", type=str, default="df")
    # Enable latent diffusion model (needed)
    # If enabled, the model will use latent diffusion.
    parser.add_argument("--latent", "-l", default=True, action="store_true")
    # Generated image format
    # Recommend to use png for better generation quality.
    # Option: jpg/png
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    # --------------------------------------------------------------------------------------------------
    # Number of generation images (required)
    # if class name is `-1` and conditional `is` True, the model would output one image per class.
    parser.add_argument("--num_images", type=int, default=1)
    # --------------------------------------------------------------------------------------------------
    # Use ema model
    # If set to false, the pt file of the ordinary model will be used
    # If true, the pt file of the ema model will be used
    parser.add_argument("--use_ema", default=True, action="store_true")
    # Weight path (required)
    parser.add_argument("--weight_path", type=str, default="D:\\code\\Integrated-Design-Diffusion-Model-main\\results_LDM-3\\df\\ckpt_last.pt")
    # Set the autoencoder checkpoint path (needed)
    parser.add_argument("--autoencoder_ckpt", type=str, default="D:\\code\\Integrated-Design-Diffusion-Model-main\\results\\autoencoder\\ckpt_last.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="D:\\code\\Integrated-Design-Diffusion-Model-main\\results_LDM-4\\vis")
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm', 'ddim' or 'plms'.
    # Option: ddpm/ddim/plms
    parser.add_argument("--sample", type=str, default="ddpm", choices=sample_choices)
    # Input image size (required)
    # [Warn] Compatible with older versions
    # [Warn] Version <= 1.1.1 need to be equal to model's image size, version > 1.1.1 can set whatever you want
    parser.add_argument("--image_size", "-i", type=parse_image_size_type, default=64)
    parser.add_argument("--conditional", default=True, action="store_true")
    parser.add_argument("--network", type=str, default="unet", choices=network_choices)
    parser.add_argument("--act", type=str, default="gelu", choices=act_choices)
    parser.add_argument("--num_classes", type=int, default=1)

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)
    # Set the use GPU in generate (required)
    parser.add_argument("--use_gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    # Init generate args
    args = init_generate_args()
    # Get version banner
    # get_version_banner()
    gen_model = Generator(gen_args=args, deploy=False)
    # for i in range(2):
    #     gen_model.generate(index=i)
    import glob
    paths = glob.glob("D:\\code\\Diffusion-Models-pytorch-main\\Image\\0\\*.png")
    for i in range(10328,len(paths)):
        gen_model.generate(path=paths[i], num_points=30, index=i)
        # print(i,paths[i])




