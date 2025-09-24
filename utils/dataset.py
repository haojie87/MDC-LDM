
import torch
import torchvision

from torch.utils.data import DataLoader, DistributedSampler, Dataset
from typing import Union
import sys
sys.path.append("D:\\code\\Integrated-Design-Diffusion-Model-main")
from iddm.config.setting import RANDOM_RESIZED_CROP_SCALE, MEAN, STD
from iddm.utils.check import check_path_is_exist

import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
import torch.nn.functional as F
from tqdm import tqdm


def get_value(image_size: Union[int, list, tuple]):
    if isinstance(image_size, int):
        return image_size
    elif isinstance(image_size, (list, tuple)):
        return image_size[0]  # 取第一个
    else:
        raise TypeError("image_size 必须是 int, list, 或 tuple")


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


def Points_Radiation(path, image_size=512, num_points=100, dis_ratio=0.2):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
    return img_points_radiation


class TUDataset(Dataset):
    def __init__(self, image_size, transforms_img, imgFile="", end_with="png" ,num_points=100, dis_ratio=0.1):
        imgs = glob.glob(os.path.join(imgFile,'*'+end_with))
        self.imgs = sorted(imgs)
        self.len = len(self.imgs)
        self.image_size = image_size
        self.num_points = num_points
        self.dis_ratio = dis_ratio
        self.tensor_transform = transforms_img
        self.img_points_radiation_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
                ])

    def __getitem__(self,index):
        idx = index % self.len
        img = Image.open(self.imgs[idx])
        if img.mode == 'RGBA':
            r, g, b, a = img.split() # 将图像的通道分离
            img = Image.merge("RGB", (r, g, b)) 
        img = np.array(img)
        img = Image.fromarray(img)
        img = self.tensor_transform(img)
        img_points_radiation = Points_Radiation(self.imgs[idx], image_size=self.image_size, num_points=self.num_points, dis_ratio=self.dis_ratio)
        img_points_radiation = self.img_points_radiation_transform(img_points_radiation)
        # return {"img":img, "img_points_radiation":img_points_radiation}
        return img, img_points_radiation

    def __len__(self):
        return len(self.imgs)


def get_dataset(image_size: Union[int, list, tuple], dataset_path=None, batch_size=2, num_workers=0, distributed=False):
    """
    Get dataset

    :param image_size: Image size
    :param dataset_path: Dataset path
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param distributed: Whether to distribute training
    :return: dataloader
    """
    check_path_is_exist(path=dataset_path)
    # Data augmentation
    transforms = torchvision.transforms.Compose([
        # Resize input size, input type is (height, width)
        # torchvision.transforms.Resize(), image_size + 1/4 * image_size
        # torchvision.transforms.Resize(size=set_resize_images_size(image_size=image_size, divisor=4
        #     )),
        torchvision.transforms.Resize(size=image_size),
        # Random adjustment cropping
        # torchvision.transforms.RandomResizedCrop(size=image_size, scale=RANDOM_RESIZED_CROP_SCALE),
        # To Tensor Format
        torchvision.transforms.ToTensor(),
        # For standardization, the mean and standard deviation
        # Refer to the initialization of ImageNet
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])
    # Load the folder data under the current path,
    # and automatically divide the labels according to the dataset under each file name
    # dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transforms)
    # image_size_ = get_value(image_size=set_resize_images_size(image_size=image_size, divisor=4))
    image_size_ = get_value(image_size)

    dataset = TUDataset(image_size=image_size_, transforms_img=transforms, imgFile=dataset_path, 
        end_with="png" ,num_points=100, dis_ratio=0.1)
    if distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True)

    return dataloader


def set_resize_images_size(image_size: Union[int, list, tuple], divisor=4):
    """
    Set resized image size
    :param image_size: Image size
    :param divisor: Divisor
    :return: image_size
    """
    if isinstance(image_size, (int, list, tuple)):
        if type(image_size) is int:
            image_size = int(image_size + image_size / divisor)
        elif type(image_size) is list:
            image_size = [int(x + x / divisor) for x in image_size]
        else:
            image_size = tuple([int(x + x / divisor) for x in image_size])
        return image_size
    else:
        raise TypeError("image_size must be int, list or tuple.")


def post_image(images, device="cpu"):
    """
    Post images
    :param images: Images
    :param device: CPU or GPU
    :return: new_images
    """
    mean_tensor = torch.tensor(data=MEAN).view(1, -1, 1, 1).to(device)
    std_tensor = torch.tensor(data=STD).view(1, -1, 1, 1).to(device)
    new_images = images * std_tensor + mean_tensor
    # Limit the image between 0 and 1
    new_images = (new_images.clamp(0, 1) * 255).to(torch.uint8)
    return new_images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "D:\\code\\Diffusion-Models-pytorch-main\\Image_val\\0"
    dataloader = get_dataset(image_size=512, dataset_path=path,
                                  batch_size=1, num_workers=0,
                                  distributed=False)
    # Number of dataset batches in the dataloader
    len_dataloader = len(dataloader)
    pbar = tqdm(dataloader)
    # Initialize images and labels
    images, img_points_radiation, loss_list = None, None, []
    for i, (images, img_points_radiation) in enumerate(pbar):
        # The images are all resized in dataloader
        # images = images.to('cuda')
        print()
        print("------------------------")
        print(images.shape,img_points_radiation.shape)
        print("------------------------")
        print() 
