import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import pydicom
from skimage.transform import resize

def display_arr_stats(arr):
    shape, vmin, vmax, vmean, vstd = arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


class ImageDataset_3D(Dataset):

    def __init__(self, img_path, img_dim):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.load(img_path)['data']  # [C, H, W]

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[center_idx-num_slice:center_idx+num_slice, :, :]
        im_size = image.shape
        print(image.shape, center_idx, num_slice)

        # Complete 3D input image as a squared x-y image
        if not(im_size[1] == im_size[2]):
            zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y plane
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)

        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1



# class ImageDataset_2D(Dataset):

#     def __init__(self, img_path, img_dim, img_slice):
#         '''
#         img_dim: new image size [h, w]
#         '''
#         self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
#         image = np.load(img_path)['data']
#         image = image[img_slice, :, :]  # Choose one slice as 2D CT image
#         imsize = image.shape

#         # Complete as a squared image
#         if not(imsize[0] == imsize[1]):
#             zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1])/2)])
#             image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

#         # Interpolate image to predefined size
#         image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR) 

#         # Scaling normalization
#         image = image / np.max(image)
#         self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
#         display_tensor_stats(self.img)

        
#     def __getitem__(self, idx):
#         grid = create_grid(*self.img_dim)
#         return grid, self.img

#     def __len__(self):
#         return 1
    


class ImageDataset_2D(Dataset):

    def __init__(self, img_path, img_dim, img_slice):
        '''
        img_dim: new image size [h, w]
        '''
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = pydicom.read_file(img_path).pixel_array
        image = np.float32(image)
        image[image > 3000] = 0
        image = image / 3000
        image = resize(image, (256, 256))
        image[image < 0] = 0
        imsize = image.shape

        # Complete as a squared image
        if not(imsize[0] == imsize[1]):
            zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

        # Interpolate image to predefined size

        # Scaling normalization
        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
        display_tensor_stats(self.img)

        
    def __getitem__(self, idx):
        grid = create_grid(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1


class ImageDataset(Dataset):

    def __init__(self, img_path, img_dim):
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        h, w = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255
        grid = create_grid(*self.img_dim[::-1])

        return grid, torch.tensor(image, dtype=torch.float32)[:, :, None]

    def __len__(self):
        return 1


class ImageDataset_new(Dataset):
    def __init__(self, data, size=100, num_samples=2**18, split='train'):
        super().__init__()
        
        # 确保数据是灰度图（即单通道）
        if len(data.shape) == 2:  # 如果是二维数组，添加一个通道维度
            self.data = data[:, :, np.newaxis]
        else:
            self.data = data

        self.img_wh = (self.data.shape[0], self.data.shape[1])
        self.img_shape = torch.tensor([self.img_wh[0], self.img_wh[1]]).float()

        print(f"[INFO] image: {self.data.shape}")

        self.num_samples = num_samples
        self.split = split
        self.size = size

        if self.split.startswith("test"):
            half_dx =  0.5 / self.img_wh[0]
            half_dy =  0.5 / self.img_wh[1]
            xs = torch.linspace(half_dx, 1-half_dx, self.img_wh[0])
            ys = torch.linspace(half_dy, 1-half_dy, self.img_wh[1])
            xv, yv = torch.meshgrid([xs, ys], indexing="ij")

            xy = torch.stack((xv.flatten(), yv.flatten())).t()

            xy_max_num = math.ceil(xy.shape[0] / 1024.0)
            padding_delta = xy_max_num * 1024 - xy.shape[0]
            zeros_padding = torch.zeros((padding_delta, 2))
            self.xs = torch.cat([xy, zeros_padding], dim=0)


    def __len__(self):
        return self.size

    def __getitem__(self, _):
        if self.split.startswith('train'):
            xs = torch.rand([self.num_samples, 2], dtype=torch.float32)
            scaled_xs = xs * self.img_shape
            indices = scaled_xs.long()
            lerp_weights = scaled_xs - indices.float()

            # 对于灰度图像，只处理单个通道
            x0 = indices[:, 0].clamp(min=0, max=self.img_wh[0]-1).long()
            y0 = indices[:, 1].clamp(min=0, max=self.img_wh[1]-1).long()
            x1 = (x0 + 1).clamp(min=0, max=self.img_wh[0]-1).long()
            y1 = (y0 + 1).clamp(min=0, max=self.img_wh[1]-1).long()

            # 双线性插值
            greyscales = self.data[x0, y0] * (1.0 - lerp_weights[:, 0]) * (1.0 - lerp_weights[:, 1]) + \
                         self.data[x0, y1] * (1.0 - lerp_weights[:, 0]) * lerp_weights[:, 1] + \
                         self.data[x1, y0] * lerp_weights[:, 0] * (1.0 - lerp_weights[:, 1]) + \
                         self.data[x1, y1] * lerp_weights[:, 0] * lerp_weights[:, 1]
        else:
            xs = self.xs
            greyscales = self.data

        results = {
            'points': xs,
            'greyscales': greyscales.squeeze(),  # 移除通道维度
        }

        return results

