import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from unet import UNet
from data_loading import BasicDataset

# ========== 关键修改1：设置Matplotlib非交互式后端（适配无图形界面） ==========
plt.switch_backend('Agg')  # 禁用图形窗口，仅保存图片

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备：{device}")

def inference_sample(model, dataset, idx):
    """推理单张样本，返回原图、真实掩码、预测掩码"""
    print(f"\n开始推理样本 ID: {dataset.ids[idx]} (索引: {idx})")
    model.eval()
    sample = dataset[idx]
    image = sample['image'].unsqueeze(0).to(device)
    mask_true = sample['mask'].numpy().squeeze()

    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output) > 0.5
        pred = pred.cpu().numpy().squeeze()

    # 反归一化图像以便显示
    img_np = sample['image'].numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    print(f"样本 {dataset.ids[idx]} 推理完成，图像尺寸：{img_np.shape}")
    return img_np, mask_true, pred

def visualize(img, mask_true, mask_pred, title='Sample', save_path='inference_results'):
    """可视化并保存图片（替换plt.show()为保存）"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(mask_true, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    axes[2].imshow(mask_pred, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    plt.suptitle(title)
    
    # 保存图片（替换plt.show()）
    save_file = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    plt.close(fig)  # 关闭画布，释放内存
    print(f"可视化结果已保存到：{save_file}")

if __name__ == '__main__':
    # ========== 关键修改2：添加路径合法性检查 ==========
    print("===== 开始执行推理 =====")
    # 检查config中的路径是否存在
    assert hasattr(config, 'model_path'), "config.py中未定义model_path"
    assert Path(config.model_path).exists(), f"模型文件不存在：{config.model_path}"
    assert Path(config.X_path).exists(), f"图像路径不存在：{config.X_path}"
    assert Path(config.y_path).exists(), f"掩码路径不存在：{config.y_path}"
    print(f"模型路径：{config.model_path}")
    print(f"图像数据集路径：{config.X_path}")
    print(f"掩码数据集路径：{config.y_path}")

    # 加载模型
    print("加载模型中...")
    model = UNet(n_channels=3, n_classes=config.n_classes).to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    print("模型加载完成！")

    # 加载数据集
    print("加载数据集...")
    dataset = BasicDataset(config.X_path, config.y_path, config.img_scale)
    print(f"数据集加载完成，共 {len(dataset)} 个样本")

    # 随机抽取两张图片
    print("随机选择样本进行推理...")
    indices = random.sample(range(len(dataset)), 2)
    for i, idx in enumerate(indices):
        img, true_mask, pred_mask = inference_sample(model, dataset, idx)
        visualize(img, true_mask, pred_mask, title=f'Sample {i+1} (ID: {dataset.ids[idx]})')
    
    print("\n===== 推理完成！所有结果已保存到 inference_results 文件夹 =====")