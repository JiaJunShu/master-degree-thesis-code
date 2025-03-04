import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# 定义模型
class model_3round(nn.Module):
    def __init__(self):
        super(model_3round, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        # 第一个隐藏层
        self.fc2 = nn.Linear(32, 64)
        # 新增的隐藏层
        self.fc3 = nn.Linear(64, 32)
        # 第二个隐藏层
        self.fc4 = nn.Linear(32, 16)
        # 输出层
        self.fc5 = nn.Linear(16, 6)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return x



# 加载标准化器
scaler = joblib.load('scaler_3round.pkl')

# 加载模型
model = model_3round()
model.load_state_dict(torch.load('model_3round.pth'))
model.eval()
'''
# 初始参数
base_data = [0.4, 1200000, -3, -3]

# 存储所有的预测结果
predictions = []

with torch.no_grad():
    for i in range(181):  # 从0到180，共181个数
        # 将最后一个参数设置为i
        new_data = base_data + [i]

        # 数据缩放
        new_data_scaled = scaler.transform([new_data])

        # 转换为tensor
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        # 执行预测
        prediction = model(new_data_tensor)

        # 将预测结果存储起来
        predictions.append(prediction.numpy().tolist())  # 假设预测结果是张量

# 将预测结果写入txt文件
with open('parameter_predict.txt', 'w') as f:
    for pred in predictions:
        # 将每一个8维向量写入文件，每个值以空格分隔
        f.write(' '.join(map(str, pred)) + '\n')
'''

# 假设你要测试的数据输入（请根据实际情况替换数值）
new_data = [[0.8,1800000.0,-4.0,7.0,0]]  # 这里是一个示例数据点，使用二维列表


# 将输入数据转换为 NumPy 数组并进行标准化
new_data_scaled = scaler.transform(new_data)  # 使用之前训练时的scaler进行标准化

# 将数据转换为 PyTorch 张量
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():  # 在评估时不需要计算梯度
    prediction = model(new_data_tensor)

# 输出预测结果
print(f'预测结果: {prediction.numpy()}')  # 将张量转换为 NumPy 数组进行打印