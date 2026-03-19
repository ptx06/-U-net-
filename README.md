# UNet图像分割项目

这是一个基于PyTorch实现的UNet图像分割项目，支持语义分割任务。

## 项目结构

```
project/
 ├── config.py              # 配置文件
 ├── data_loading.py        # 数据加载模块
 ├── utils.py               # 工具函数
 ├── train.py              # 训练脚本
 ├── val.py                # 验证脚本
 ├── inference.py          # 推理脚本
 ├── application.py        # 应用脚本（GUI/CLI）
 ├── requirements.txt      # 依赖包列表
 ├── unet/                 # UNet模型模块
 │   ├── __init__.py
 │   ├── unet_parts.py     # 模型组件
 │   └── unet_model.py     # 完整模型
 ├── data/                 # 数据目录
 │   ├── image/           # 原始图像
 │   └── matte/           # 标签图像
 └── savemodel/           # 模型保存目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将您的训练数据放入以下目录：
- `data/image/` - 原始图像文件
- `data/matte/` - 对应的标签图像（掩码）

**注意**: 图像和掩码文件应该一一对应，且文件名相同。

### 3. 配置参数

编辑 `config.py` 文件，根据需要调整参数：
- 数据路径
- 模型参数
- 训练参数
- 图像尺寸等

### 4. 训练模型

```bash
python train.py
```

### 5. 验证模型

```bash
python val.py
```

### 6. 使用模型进行推理

#### 命令行界面：
```bash
python inference.py
```

#### GUI界面：
```bash
python application.py
```

## 详细说明

### 配置文件 (config.py)

包含所有可配置的参数：
- 数据路径设置
- 模型架构参数
- 训练超参数
- 设备配置等

### 数据加载 (data_loading.py)

提供数据加载和预处理功能：
- 自定义数据集类
- 数据增强
- 自动划分训练/验证集

### UNet模型 (unet/)

完整的UNet实现：
- `unet_parts.py`: 模型组件（双卷积、下采样、上采样等）
- `unet_model.py`: 完整的UNet模型架构

### 训练脚本 (train.py)

训练功能包括：
- 自动设备检测（CPU/GPU）
- 训练过程监控
- 模型保存
- TensorBoard日志记录
- 学习率调度

### 验证脚本 (val.py)

验证功能包括：
- 模型性能评估
- IoU和Dice系数计算
- 结果可视化
- 不同阈值性能分析

### 推理脚本 (inference.py)

推理功能包括：
- 单张图像预测
- 批量图像预测
- 结果可视化
- 预测结果保存

### 应用脚本 (application.py)

提供用户友好的界面：
- **GUI模式**: 图形界面，支持图像加载、实时分割、结果保存
- **CLI模式**: 命令行界面，适合批量处理

## 使用示例

### 训练新模型

1. 准备训练数据
2. 调整 `config.py` 中的参数
3. 运行 `python train.py`

### 使用预训练模型

1. 将模型文件放入 `savemodel/` 目录
2. 运行 `python inference.py` 或 `python application.py`

### 自定义模型

修改 `unet/unet_model.py` 中的模型架构，然后重新训练。

## 性能指标

项目支持以下评估指标：
- **IoU (Intersection over Union)**: 交并比
- **Dice系数**: 相似度度量
- **损失曲线**: 训练和验证损失

## 注意事项

1. **数据准备**: 确保图像和掩码文件一一对应
2. **图像格式**: 支持常见图像格式（PNG, JPG, JPEG等）
3. **GPU支持**: 自动检测并使用GPU（如果可用）
4. **模型保存**: 最佳模型自动保存到 `savemodel/` 目录

## 扩展功能

项目可以轻松扩展以下功能：
- 多类别分割
- 不同的损失函数
- 额外的数据增强
- 模型集成
- Web服务接口

## 故障排除

### 常见问题

1. **内存不足**: 减小 `config.py` 中的 `BATCH_SIZE`
2. **模型不收敛**: 调整学习率或使用预训练权重
3. **文件不存在**: 检查数据路径配置

### 获取帮助

如有问题，请检查：
- 依赖包是否安装完整
- 数据路径是否正确
- 文件权限是否足够