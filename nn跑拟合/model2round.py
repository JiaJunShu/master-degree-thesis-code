import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 读取数据
data = pd.read_csv('output_file.csv',header=1,encoding='gbk')

# 获取特征X（前四列）
X = data.iloc[:, :4].values  # 选择前四列作为输入特征

# 获取标签y（第五至第十二列）
y = data.iloc[:, 4:12].values  # 选择第五至第十二列作为输出标签

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_2round.pkl')  # 保存 scaler

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # 转换为 (n_samples, 3)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # 如果有需要的话

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

# 初始化模型、损失函数和优化器
model = model_2round()
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.005)

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
torch.save(model.state_dict(), 'model_2round.pth')



