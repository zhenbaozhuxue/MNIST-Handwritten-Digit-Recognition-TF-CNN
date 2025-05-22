import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 直接导入已有的数据加载模块
from src.data_loader import load_and_preprocess_data


def evaluate_model(model_path, x_test, y_test):
    """评估模型性能并生成报告"""
    model = load_model(model_path)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    accuracy = np.mean(y_pred == y_test)
    print(f"测试准确率: {accuracy:.4f}")

    report = classification_report(y_test, y_pred)
    print("\n分类报告:")
    print(report)

    return y_pred, accuracy


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)
    plt.savefig(save_path)
    print(f"混淆矩阵已保存至: {save_path}")


if __name__ == "__main__":
    # 加载测试数据（只需要测试集部分）
    _, _, x_test, y_test = load_and_preprocess_data(download_if_missing=False)

    # 模型路径
    model_dir = "results/models"
    best_model_path = os.path.join(model_dir, "best_model.keras")
    final_model_path = os.path.join(model_dir, "final_model.keras")

    # 评估最佳模型
    print("\n=== 评估最佳模型 ===")
    y_pred_best, accuracy_best = evaluate_model(best_model_path, x_test, y_test)
    plot_confusion_matrix(y_test, y_pred_best,
                          "最佳模型混淆矩阵",
                          os.path.join("results", "confusion_matrix_best.png"))

    # 评估最终模型（可选）
    print("\n=== 评估最终模型 ===")
    y_pred_final, accuracy_final = evaluate_model(final_model_path, x_test, y_test)
    plot_confusion_matrix(y_test, y_pred_final,
                          "最终模型混淆矩阵",
                          os.path.join("results", "confusion_matrix_final.png"))

    # 对比结果
    print("\n=== 模型对比 ===")
    print(f"最佳模型准确率: {accuracy_best:.4f}")
    print(f"最终模型准确率: {accuracy_final:.4f}")
    print(f"{'选择最佳模型' if accuracy_best > accuracy_final else '两个模型性能相同'}")