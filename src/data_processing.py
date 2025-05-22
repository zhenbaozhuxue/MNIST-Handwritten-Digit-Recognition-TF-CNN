# data_processing.py - 只负责数据预处理

def preprocess_cnn(x_train, x_test):
    """为CNN模型预处理数据"""
    # 重塑数据以适应CNN (样本数, 高度, 宽度, 通道数)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    return x_train, x_test


