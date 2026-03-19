import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import config

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        
        # 使用固定尺寸而不是比例缩放，确保所有图像尺寸一致
        if scale < 1.0:
            # 如果使用比例缩放，计算新尺寸
            newW, newH = int(scale * w), int(scale * h)
        else:
            # 使用固定尺寸（例如160x160）
            newW, newH = 160, 160
            
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        # 对掩膜使用最近邻插值，对原图使用双三次插值
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        
        # 如果是RGBA图像（4通道），转换为RGB（3通道）
        if pil_img.mode == 'RGBA' and not is_mask:
            pil_img = pil_img.convert('RGB')
        
        img_ndarray = np.asarray(pil_img)

        # ========== 新增处理：若掩膜是彩色图（3通道），转换为单通道（取第一个通道） ==========
        if is_mask and img_ndarray.ndim == 3:
            # 假设彩色掩膜三个通道值相同，取第一个通道即可
            img_ndarray = img_ndarray[:, :, 0]

        if not is_mask:
            if img_ndarray.ndim == 2:
                # 灰度图转为 (C, H, W)
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                # 彩色图 (H, W, C) -> (C, H, W)
                img_ndarray = img_ndarray.transpose((2, 0, 1))
                
                # 如果通道数不是3，转换为3通道
                if img_ndarray.shape[0] == 4:  # RGBA -> RGB
                    img_ndarray = img_ndarray[:3, :, :]
                elif img_ndarray.shape[0] == 1:  # 单通道 -> 3通道
                    img_ndarray = np.repeat(img_ndarray, 3, axis=0)

        img_ndarray = img_ndarray / 255.0
        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for ID {name}: {mask_file}'

        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        assert img.size == mask.size, f'Image and mask size mismatch for ID {name}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # 确保掩膜有通道维度
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().float().contiguous()
        }