import numpy as np

def softmax(x):
    """
    Softmax函数: 将logits转换为概率分布
    """
    # 数值稳定性 - 减去最大值
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    计算交叉熵损失
    y_pred: 模型预测概率
    y_true: 真实标签
    """
    samples = len(y_pred)
    
    # 防止log(0)错误 - 限制概率范围
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # 只计算正确类别的log概率
    correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
    
    # Loss = -log(p)
    negative_log_likelihoods = -np.log(correct_confidences)
    
    return np.mean(negative_log_likelihoods)

def cross_entropy_gradient(y_pred, y_true):
    """
    计算Softmax+CrossEntropy的联合梯度
    """
    samples = len(y_pred)
    
    # 简化的梯度公式: dL/dZ = P - Y
    return (y_pred - y_true) / samples