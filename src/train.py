# train.py - 只负责模型训练和断点续训
import os
import argparse
import json
from model import create_cnn_model  # 从外部导入模型定义
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_loader import load_and_preprocess_data  # 假设已有数据加载模块


def train_model(initial_epoch=0, total_epochs=20, model_save_dir="results/models"):
    """训练模型，支持断点续训

    Args:
        initial_epoch: 从第几轮开始训练（用于断点续训）
        total_epochs: 总共训练的轮数
        model_save_dir: 模型保存目录
    """
    # 确保保存目录存在
    os.makedirs(model_save_dir, exist_ok=True)

    # 加载数据
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # 检查是否需要加载已有模型
    if initial_epoch > 0:
        model_path = os.path.join(model_save_dir, f"model_epoch_{initial_epoch}.keras")
        if os.path.exists(model_path):
            print(f"从第 {initial_epoch + 1} 轮开始继续训练，加载模型: {model_path}")
            model = load_model(model_path)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    else:
        print("从头开始训练新模型")
        model = create_cnn_model()  # 使用外部定义的模型

    # 设置回调函数
    callbacks = [
        # 保存最佳模型
        ModelCheckpoint(
            os.path.join(model_save_dir, "best_model.keras"),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        # 早停机制
        EarlyStopping(
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # 训练模型
    print(f"开始训练: 从第 {initial_epoch + 1} 轮到第 {total_epochs} 轮")
    history = model.fit(
        x_train, y_train,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, f"model_epoch_{total_epochs}.keras")
    model.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    # 保存训练历史
    history_path = os.path.join(model_save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"训练历史已保存至: {history_path}")

    # 在测试集上评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集准确率: {test_acc:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练手写数字识别模型')
    parser.add_argument('--initial_epoch', type=int, default=10,
                        help='从第几轮开始训练（0表示从头开始）')
    parser.add_argument('--total_epochs', type=int, default=20,
                        help='总共训练的轮数')
    parser.add_argument('--model_dir', type=str, default="results/models",
                        help='模型保存目录')

    args = parser.parse_args()

    train_model(
        initial_epoch=args.initial_epoch,
        total_epochs=args.total_epochs,
        model_save_dir=args.model_dir
    )