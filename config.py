# 数据路径
X_path = './data/image'          # 原始图像文件夹
y_path = './data/matte'          # 标签掩膜文件夹
model_path = './savemodel/unet_model.ckpt'   # 模型保存路径

# 预处理参数
img_size = (160, 160)            # 仅作参考，实际缩放由img_scale控制
img_scale = 0.1                   # 缩放比例（例如原图1000x1000 -> 100x100）
batch_size = 1
num_workers = 0

# 模型参数
n_classes = 1                     # 二分类（人像/背景）
use_dice = False                  # 是否使用Dice Loss（PPT中dice=None）

# 训练参数
lr = 1e-4
opt = 'SGD'                       # 可选 'SGD' 或 'RMSprop'
momentum = 0.99
weight_decay = 1e-8
epochs = 50