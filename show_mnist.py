import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def show_training_data():
    # 加载数据
    (x_train, y_train), _ = mnist.load_data()
    
    print(f"训练集总共有 {len(x_train)} 张图片")
    print(f"每一张图片的大小是: {x_train[0].shape}")

    # 设置画布
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    fig.suptitle('MNIST Training Data Samples (What your model learned)', fontsize=16)

    # 随机抽取32张图片展示
    indices = np.random.randint(0, len(x_train), 32)
    
    for ax, idx in zip(axes.flat, indices):
        # 显示图片（灰度图）
        ax.imshow(x_train[idx], cmap='gray')
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 显示标签
        ax.set_title(f"Label: {y_train[idx]}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_training_data()