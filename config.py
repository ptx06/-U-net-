# 数据路径
X_path = './data/image'          # 原始图像文件夹
y_path = './data/matte'          # 标签掩膜文件夹
model_path = './savemodel/unet_model.ckpt'   # 模型保存路径

# 预处理参数
img_size = (160, 160)
img_scale = 1.0                  # 提升分辨率，保留更多细节
batch_size = 1                   # 充分利用RTX 4090显存
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
