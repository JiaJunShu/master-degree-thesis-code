import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler


scaler= joblib.load('scaler_2round.pkl')

# 定义全连接神经网络
class model_2round(nn.Module):
    def __init__(self):
        super(model_2round, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 输入 4 个特征
        self.fc2 = nn.Linear(16, 8)   # 隐藏层
        self.fc3 = nn.Linear(8, 8)    # 输出层，输出 3 个值

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型实例
model = model_2round()

# 加载模型的状态字典
model.load_state_dict(torch.load('model_2round.pth'))

# 切换到评估模式
model.eval()

# 假设你要测试的数据输入（请根据实际情况替换数值）
new_data = [[0.4,1200000.0,-5.0,-9.0]]  # 这里是一个示例数据点，使用二维列表


# 将输入数据转换为 NumPy 数组并进行标准化
new_data_scaled = scaler.transform(new_data)  # 使用之前训练时的scaler进行标准化

# 将数据转换为 PyTorch 张量
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():  # 在评估时不需要计算梯度
    prediction = model(new_data_tensor)

# 输出预测结果
print(f'预测结果: {prediction.numpy()}')  # 将张量转换为 NumPy 数组进行打印