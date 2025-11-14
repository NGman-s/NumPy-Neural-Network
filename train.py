import numpy as np
import time
from layers import Dense, ReLU
from loss import softmax, cross_entropy_loss, cross_entropy_gradient

# 工具函数
def load_mnist_data():
    """加载并预处理数据"""
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    print("正在加载 MNIST 数据集...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 归一化处理
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    # One-hot编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

# 网络架构配置
network = [
    Dense(784, 128),
    ReLU(),
    Dense(128, 10)
]

# 训练循环
def train(x_train, y_train, epochs=20, learning_rate=0.1, batch_size=64):
    print(f"开始训练... (架构: 784->128->10 | 学习率: {learning_rate})")
    
    for epoch in range(epochs):
        loss = 0
        start_time = time.time()
        
        # Batch迭代
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # 前向传播
            output = x_batch
            for layer in network:
                output = layer.forward(output)
            
            # 计算概率和损失
            predictions = softmax(output)
            loss += cross_entropy_loss(predictions, y_batch)

            # 反向传播
            grad = cross_entropy_gradient(predictions, y_batch)
            
            # 传递梯度
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # 计算平均损失
        loss /= (len(x_train) // batch_size)
        print(f"Epoch {epoch+1}/{epochs}\tLoss: {loss:.4f}\tTime: {time.time()-start_time:.2f}s")

# 评估与测试
def evaluate(x_test, y_test):
    print("\n正在评估测试集准确率...")
    correct = 0
    total = 0
    
    # 分批预测
    for i in range(0, len(x_test), 1000):
        x_batch = x_test[i:i+1000]
        y_batch = y_test[i:i+1000]
        
        output = x_batch
        for layer in network:
            output = layer.forward(output)
            
        predictions = np.argmax(softmax(output), axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        
        correct += np.sum(predictions == true_labels)
        total += len(predictions)
        
    accuracy = correct / total * 100
    print(f"最终测试集准确率: {accuracy:.2f}%")
    return accuracy

def save_model(network, filename="model.npz"):
    """保存模型参数到文件"""
    print(f"正在保存模型到 {filename} ...")
    parameters = {}
    for i, layer in enumerate(network):
        # 只保存全连接层的参数
        if hasattr(layer, 'weights'):
            parameters[f'w_{i}'] = layer.weights
            parameters[f'b_{i}'] = layer.bias
    
    # 保存参数
    np.savez(filename, **parameters)
    print("保存成功！")

# 主程序入口
if __name__ == "__main__":
    # 1. 加载数据
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # 2. 开始训练
    train(x_train, y_train, epochs=20, learning_rate=0.1)
    
    # 3. 最终测试
    evaluate(x_test, y_test)

    # 4. 保存模型
    save_model(network, "model.npz")