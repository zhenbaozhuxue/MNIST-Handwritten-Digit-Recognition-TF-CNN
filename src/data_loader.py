# src/data_loader.py
import os
import tensorflow as tf
import numpy as np

# 定义项目内的数据存储路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database')
MNIST_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
LOCAL_PATH = os.path.join(DATA_DIR, 'mnist.npz')


def load_and_preprocess_data(download_if_missing=True):
    """加载并预处理数据，根据参数决定是否自动下载"""
    os.makedirs(DATA_DIR, exist_ok=True)  # 确保database文件夹存在

    # 检查数据是否存在
    if not os.path.exists(LOCAL_PATH):
        if download_if_missing:
            print(f"下载MNIST数据集到 {LOCAL_PATH}...")
            # 使用Keras下载并保存到指定路径
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            np.savez(LOCAL_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            print("下载完成")
        else:
            raise FileNotFoundError(f"数据文件不存在: {LOCAL_PATH}")

    # 加载本地数据
    with np.load(LOCAL_PATH, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    # 数据预处理
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test