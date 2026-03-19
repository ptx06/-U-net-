import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from unet import UNet
from data_loading import BasicDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def inference_sample(model, dataset, idx):
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
    return img_np, mask_true, pred

def visualize(img, mask_true, mask_pred, title='Sample'):
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
    plt.show()

if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=config.n_classes).to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    dataset = BasicDataset(config.X_path, config.y_path, config.img_scale)

    # 随机抽取两张图片（一张训练集一张测试集，但这里未划分，故随机选两个）
    indices = random.sample(range(len(dataset)), 2)
    for i, idx in enumerate(indices):
        img, true_mask, pred_mask = inference_sample(model, dataset, idx)
        visualize(img, true_mask, pred_mask, title=f'Sample {i+1} (ID: {dataset.ids[idx]})')