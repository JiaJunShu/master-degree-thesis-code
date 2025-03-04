import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# 读取数据
data = pd.read_csv('my_dataout(2).csv')

# 分离特征和标签
X = data.iloc[:, :4].values  # 选择前四列作为输入特征
y = data.iloc[:, 4:6].values  # 选择第五至第七列作为输出标签（形状为 (n_samples, 3)）

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_clr.pkl')  # 保存 scaler

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # 转换为 (n_samples, 3)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # 如果有需要的话

# 定义全连接神经网络
class model_Clr(nn.Module):
    def __init__(self):
        super(model_Clr, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 输入 4 个特征
        self.fc2 = nn.Linear(16, 8)   # 隐藏层
        self.fc3 = nn.Linear(8, 2)    # 输出层，输出 3 个值

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = model_Clr()
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    model.train()

    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i + batch_size]
        labels = y_train_tensor[i:i + batch_size]

        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('训练完成')

# 保存模型参数到 'model_Cy.pth' 文件
torch.save(model.state_dict(), 'model_Clr.pth')

# 创建模型实例
model = model_Clr()

# 加载模型的状态字典
model.load_state_dict(torch.load('model_Clr.pth'))

# 切换到评估模式
model.eval()

# 假设你要测试的数据输入（请根据实际情况替换数值）
new_data = [[0.9,120000000,0.1,4]]  # 这里是一个示例数据点，ma re beta alpha
#输出 my mz

# 将输入数据转换为 NumPy 数组并进行标准化
new_data_scaled = scaler.transform(new_data)  # 使用之前训练时的scaler进行标准化

# 将数据转换为 PyTorch 张量
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():  # 在评估时不需要计算梯度
    prediction = model(new_data_tensor)

# 输出预测结果
print(f'预测结果: {prediction.numpy()}')  # 将张量转换为 NumPy 数组进行打印