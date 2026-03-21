import os
# 修复OMP_NUM_THREADS警告
os.environ['OMP_NUM_THREADS'] = '1'

import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from unet import UNet
from data_loading import BasicDataset
from torch.utils.data import DataLoader

# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# 模型
unet_model = UNet(n_channels=3, n_classes=config.n_classes).to(device)

# 损失函数
criterion = nn.BCEWithLogitsLoss()

# Dice Loss（如果启用）
if config.use_dice:
    from utils import BinaryDiceLoss
    dice_loss_func = BinaryDiceLoss()
else:
    dice_loss_func = None

# 优化器
if config.opt == 'SGD':
    optimizer = optim.SGD(unet_model.parameters(), lr=config.lr,
                          momentum=config.momentum, weight_decay=config.weight_decay)
elif config.opt == 'RMSprop':
    optimizer = optim.RMSprop(unet_model.parameters(), lr=config.lr,
                              weight_decay=config.weight_decay, momentum=config.momentum)
else:
    raise ValueError(f'Unsupported optimizer: {config.opt}')

# TensorBoard
writer = SummaryWriter(log_dir='./runs/unet_experiment')

# 数据
dir_img = Path(config.X_path)
dir_mask = Path(config.y_path)
dataset = BasicDataset(dir_img, dir_mask, config.img_scale)
train_loader = DataLoader(dataset, batch_size=config.batch_size,
                          shuffle=True, num_workers=config.num_workers,
                          pin_memory=False, persistent_workers=False)

global_step = 0

def train_one_batch(data, model, criterion, optimizer, dice=None):
    global global_step
    inputs = data['image'].to(device)
    labels = data['mask'].to(device, dtype=torch.float)

    optimizer.zero_grad()
    outputs = model(inputs)

    if dice is not None:
        dice_loss = dice(outputs, labels)
        loss = criterion(outputs, labels) + dice_loss
    else:
        loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    global_step += 1
    return outputs, loss

def train_one_epoch(loader, model, criterion, optimizer, epoch, dice=None):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    running_loss = 0.0

    with tqdm(total=num_batches, desc=f'Epoch {epoch+1}/{config.epochs}') as pbar:
        for i, data in enumerate(loader):
            outputs, loss = train_one_batch(data, model, criterion, optimizer, dice)
            total_loss += loss.item()
            running_loss += loss.item()

            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            if i % 20 == 19:
                avg_loss = running_loss / 20
                print(f'[{epoch+1}, {i+1:5d}] loss: {avg_loss:.3f}')
                # 记录图像示例
                img = data['image'][0].cpu().detach()
                pred_mask = (torch.sigmoid(outputs[0]) > 0.5).float().cpu().detach()
                true_mask = data['mask'][0].cpu().detach()
                writer.add_images('images', img.unsqueeze(0), global_step)
                writer.add_images('masks/true', true_mask.unsqueeze(0), global_step)
                writer.add_images('masks/pred', pred_mask.unsqueeze(0), global_step)
                running_loss = 0.0

            pbar.update(1)

    return total_loss / num_batches

def train(loader, model, criterion, dice, optimizer, epochs, model_path):
    best_loss = float('inf')
    for epoch in range(epochs):
        avg_loss = train_one_epoch(loader, model, criterion, optimizer, epoch, dice)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with loss {best_loss:.4f}')
    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    save_dir = os.path.dirname(config.model_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train(train_loader, unet_model, criterion, dice_loss_func, optimizer,
          config.epochs, config.model_path)