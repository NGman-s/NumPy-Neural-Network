import numpy as np

# 基类
class Layer:
    def forward(self, input):
        raise NotImplementedError
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

# 全连接层
class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        input_size: 输入特征数量
        output_size: 输出特征数量
        """
        # 权重初始化 - He Initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        
        # 偏置初始化为0
        self.bias = np.zeros((1, output_size))
        
        # 存储前向传播输入，反向传播计算梯度
        self.input = None
        
        # 存储梯度
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        """
        前向传播公式: Y = X · W + b
        """
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        反向传播
        output_gradient: 从上层传回的梯度
        """
        
        # 计算权重梯度
        weights_gradient = np.dot(self.input.T, output_gradient)
        
        # 计算输入梯度
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # 更新参数
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        
        return input_gradient

# ReLU激活函数
class ReLU(Layer):
    def forward(self, input):
        """
        ReLU 公式: f(x) = max(0, x)
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        """
        ReLU导数: x>0时导数为1，x<=0时导数为0
        """
        relu_grad = output_gradient * (self.input > 0)
        return relu_grad