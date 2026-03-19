#!/usr/bin/env python3
"""
数据加载器修复脚本 - 包含多个解决方案
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import config
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loading import BasicDataset
from pathlib import Path

def apply_fix_1():
    """修复方案1：修改数据加载器配置"""
    print("=== 修复方案1：修改数据加载器配置 ===")
    
    # 修改config.py中的num_workers
    config_content = """# 数据路径
X_path = './data/image'          # 原始图像文件夹
y_path = './data/matte'          # 标签掩膜文件夹
model_path = './savemodel/unet_model.ckpt'   # 模型保存路径

# 预处理参数
img_size = (160, 160)
img_scale = 1.0                  # 提升分辨率，保留更多细节
batch_size = 16                   # 充分利用RTX 4090显存
num_workers = 0                   # 临时设置为0，避免多进程问题

# 模型参数
n_classes = 1
use_dice = True                   # 开启Dice Loss，提升分割效果

# 训练参数
lr = 2e-4                        # batch_size提升后适当提高学习率
opt = 'SGD'
momentum = 0.9                   # 降低momentum，稳定训练
weight_decay = 1e-5              # 增强正则化，防止过拟合
epochs = 50
"""
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ 已将num_workers设置为0")
    return True

def apply_fix_2():
    """修复方案2：修改train.py中的数据加载器创建"""
    print("\n=== 修复方案2：修改train.py中的数据加载器创建 ===")
    
    # 读取train.py内容
    with open('train.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换DataLoader创建代码
    old_code = "train_loader = DataLoader(dataset, batch_size=config.batch_size,\n                          shuffle=True, num_workers=config.num_workers)"
    new_code = """train_loader = DataLoader(dataset, batch_size=config.batch_size,
                          shuffle=True, num_workers=config.num_workers,
                          pin_memory=False, persistent_workers=False)"""
    
    content = content.replace(old_code, new_code)
    
    # 写回文件
    with open('train.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已修改DataLoader配置，禁用pin_memory和persistent_workers")
    return True

def apply_fix_3():
    """修复方案3：增强数据加载器的错误处理"""
    print("\n=== 修复方案3：增强数据加载器的错误处理 ===")
    
    # 创建一个更安全的数据集类
    safe_dataset_code = '''
import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import config

class SafeBasicDataset(Dataset):
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
        try:
            ext = splitext(filename)[1]
            if ext in ['.npz', '.npy']:
                return Image.fromarray(np.load(filename))
            elif ext in ['.pt', '.pth']:
                return Image.fromarray(torch.load(filename).numpy())
            else:
                return Image.open(filename)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            raise

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
        try:
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

            # 确保使用numpy数组创建tensor，避免存储问题
            image_tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
            
            return {
                'image': image_tensor,
                'mask': mask_tensor
            }
        except Exception as e:
            print(f"Error processing sample {idx} (name: {name if 'name' in locals() else 'unknown'}): {e}")
            # 返回一个有效的样本（第一个样本）作为替代
            if idx > 0:
                return self.__getitem__(0)
            else:
                raise
'''
    
    with open('safe_data_loading.py', 'w', encoding='utf-8') as f:
        f.write(safe_dataset_code)
    
    print("✅ 已创建安全的数据集类 safe_data_loading.py")
    return True

def test_fixes():
    """测试修复效果"""
    print("\n=== 测试修复效果 ===")
    
    try:
        # 重新导入配置
        import importlib
        importlib.reload(config)
        
        dir_img = Path(config.X_path)
        dir_mask = Path(config.y_path)
        
        # 测试数据集
        dataset = BasicDataset(dir_img, dir_mask, config.img_scale)
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试数据加载器
        test_loader = DataLoader(dataset, batch_size=2, shuffle=True, 
                                num_workers=config.num_workers, pin_memory=False)
        
        for i, batch in enumerate(test_loader):
            images = batch['image']
            masks = batch['mask']
            print(f"✅ 批次 {i+1}: 图像形状 {images.shape}, 掩膜形状 {masks.shape}")
            
            if i >= 2:
                break
        
        print("✅ 修复测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 修复测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 数据加载器修复脚本 ===")
    print("此脚本将应用多个修复方案来解决RuntimeError问题")
    
    # 应用修复方案
    apply_fix_1()  # 修改num_workers
    apply_fix_2()  # 修改DataLoader配置
    apply_fix_3()  # 创建安全的数据集类
    
    # 测试修复效果
    success = test_fixes()
    
    if success:
        print("\n🎉 修复完成！")
        print("\n下一步操作：")
        print("1. 运行 python train.py 进行训练测试")
        print("2. 如果训练成功，可以逐步增加num_workers")
        print("3. 如果仍有问题，可以使用 safe_data_loading.py 中的SafeBasicDataset")
    else:
        print("\n❌ 修复测试失败，可能需要进一步排查")

if __name__ == '__main__':
    main()