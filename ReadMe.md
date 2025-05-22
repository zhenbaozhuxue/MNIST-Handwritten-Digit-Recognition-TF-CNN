# 手写数字识别项目

基于卷积神经网络的MNIST手写数字识别系统，支持断点续训和模型版本管理。

## 项目结构
handwritten_digit/
├── model.py               # 模型定义
├── train.py               # 训练脚本
├── src/                   # 源代码包
│   └── __init__.py        # 包标识
├── requirements.txt       # 依赖列表
└── README.md              # 使用说明
## 环境准备

1. **创建虚拟环境**
   ```bash
   python -m venv venv
   
   # Linux/macOS
   source venv/bin/activate
   
   # Windows
   .\venv\Scripts\activate
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 训练模型

### 从头开始训练python train.py --total_epochs 20
### 断点续训python train.py --initial_epoch 10 --total_epochs 20
## 主要参数说明

| 参数           | 描述                          | 默认值           |
|----------------|-------------------------------|------------------|
| `--initial_epoch` | 从第几轮开始训练（0表示从头开始） | 0                |
| `--total_epochs`  | 总共训练的轮数                | 20               |
| `--model_dir`     | 模型保存目录                  | results/models   |

## 模型保存策略

训练过程中会保存两种模型：
- `model_epoch_{轮数}.keras`: 按轮次保存的模型，用于断点续训
- `best_model.keras`: 验证集准确率最高的模型

## 项目特点

1. **模块化设计**：模型定义与训练逻辑完全分离
2. **断点续训**：支持从任意轮次恢复训练
3. **自动保存**：自动保存最佳模型和按轮次保存模型
4. **元数据记录**：保存完整训练历史，便于分析模型性能

## 依赖版本说明

- TensorFlow: 2.15.0
- NumPy: 1.26.0
- Matplotlib: 3.8.0
    