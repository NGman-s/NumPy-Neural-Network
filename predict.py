import numpy as np
from PIL import Image, ImageOps
from layers import Dense, ReLU
from loss import softmax

def load_model(filename="model.npz"):
    """加载保存的模型参数"""
    print(f"正在加载模型 {filename} ...")
    data = np.load(filename)
    
    # 构建网络结构 784 -> 128 -> 10
    network = [
        Dense(784, 128),
        ReLU(),
        Dense(128, 10)
    ]
    
    # 填入保存的参数
    network[0].weights = data['w_0']
    network[0].bias = data['b_0']
    network[2].weights = data['w_2']
    network[2].bias = data['b_2']
    
    return network

def preprocess_image(image_path):
    """图像预处理：针对手写数字图片优化"""
    try:
        # 打开图片并转为灰度图
        img = Image.open(image_path).convert('L')
        
        # 智能反色（白纸黑字）
        if np.mean(np.array(img)) > 128:
            img = ImageOps.invert(img)
        
        # 二值化处理，去掉背景
        img = img.point(lambda x: 255 if x > 150 else 0)

        # 获取数字包围盒并裁剪
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        # 调整大小到20x20
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        
        # 创建28x28黑色画布，数字居中
        new_img = Image.new('L', (28, 28), 0)
        new_img.paste(img, (4, 4))
        
        # 转为numpy并归一化
        img_array = np.array(new_img) / 255.0
        
        # 拉平
        img_vector = img_array.reshape(1, -1)
        
        return img_vector, new_img

    except Exception as e:
        print(f"图片处理出错: {e}")
        return None, None

def predict(network, input_vector):
    """推理过程"""
    output = input_vector
    for layer in network:
        output = layer.forward(output)
    
    # 计算概率
    probs = softmax(output)
    predicted_num = np.argmax(probs)
    confidence = probs[0][predicted_num]
    
    return predicted_num, confidence

if __name__ == "__main__":
    # 加载模型
    network = load_model("model.npz")
    
    # 指定图片路径
    image_path = "4.jpg" 
    
    # 处理图片
    input_vector, original_img = preprocess_image(image_path)
    
    if input_vector is not None:
        # 预测
        digit, conf = predict(network, input_vector)
        print("-" * 30)
        print(f"🤖 模型认为这张图是数字: 【 {digit} 】")
        print(f"📊 置信度: {conf*100:.2f}%")
        print("-" * 30)
        
        # 展示处理后的图片
        original_img.show()