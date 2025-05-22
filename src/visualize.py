# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
from data_loader import load_and_preprocess_data
from data_processing import preprocess_cnn

plt.rcParams["font.family"] = ["FangSong"]  # 设置为仿宋字体
def plot_training_history(history=None):
    """绘制训练历史曲线（支持传入历史数据或从文件加载）"""
    if history is None:
        try:
            history = np.load('results/history/training_history.npy', allow_pickle=True).item()
        except FileNotFoundError:
            print("错误: 未找到训练历史数据，请先运行 train.py")
            return

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='训练准确率')
    plt.plot(history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/history/training_history.png')
    print("训练历史图表已保存至: results/history/training_history.png")
    plt.show()  # 显示图表


def visualize_predictions():
    """可视化模型预测结果"""
    try:
        # 加载测试数据
        _, _, x_test, y_test = load_and_preprocess_data()
        _, x_test_cnn = preprocess_cnn(np.zeros((1, 28, 28)), x_test)  # 仅处理测试集

        # 加载模型
        model = load_model('results/models/final_model')

        # 预测
        predictions = model.predict(x_test_cnn)
        predicted_labels = np.argmax(predictions, axis=1)

        # 可视化随机样本
        plt.figure(figsize=(12, 8))
        indices = np.random.choice(len(x_test), 12, replace=False)

        for i, idx in enumerate(indices):
            plt.subplot(3, 4, i + 1)
            plt.imshow(x_test[idx], cmap='gray')

            is_correct = y_test[idx] == predicted_labels[idx]
            color = 'green' if is_correct else 'red'

            plt.title(f"预测: {predicted_labels[idx]}\n实际: {y_test[idx]}",
                      color=color, fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('results/predictions/prediction_samples.png')
        print("预测样本图表已保存至: results/predictions/prediction_samples.png")
        plt.show()  # 显示图表

    except FileNotFoundError:
        print("错误: 未找到模型文件，请先运行 train.py")


def main():
    """主函数：生成所有可视化结果"""
    plot_training_history()
    visualize_predictions()


if __name__ == "__main__":
    main()