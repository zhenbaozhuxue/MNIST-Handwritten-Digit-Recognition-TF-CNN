# src/validate_data.py
import warnings
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from data_loader import load_and_preprocess_data
import numpy as np

# 禁用字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# 自动检测系统中可用的中文字体
def set_chinese_font():
    # 按优先级排列的中文字体列表
    chinese_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            print(f"已设置中文字体: {font}")
            return

    print("警告: 未找到可用的中文字体，图表标题可能无法正确显示")


# 设置字体
set_chinese_font()


def visualize_sample_data(x_train, y_train, num_samples=5):
    """可视化样本数据"""
    plt.figure(figsize=(10, 4))

    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"标签: {y_train[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_data.png')
    print(f"样本数据已保存为 'sample_data.png'")


def print_data_stats(x_train, y_train, x_test, y_test):
    """打印数据统计信息"""
    print("\n数据统计信息:")
    print(f"训练样本数量: {len(x_train)}")
    print(f"测试样本数量: {len(x_test)}")
    print(f"图像尺寸: {x_train[0].shape}")
    print(f"标签范围: {np.min(y_train)}~{np.max(y_train)}")
    print(f"训练数据形状: {x_train.shape}")
    print(f"标签数据形状: {y_train.shape}")


def main():
    # 加载数据（自动下载，如果缺失）
    x_train, y_train, x_test, y_test = load_and_preprocess_data(download_if_missing=True)

    # 打印数据统计信息
    print_data_stats(x_train, y_train, x_test, y_test)

    # 可视化样本数据
    visualize_sample_data(x_train, y_train)


if __name__ == "__main__":
    main()