import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import torch.nn.functional as F
# 确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


os.chdir('D:/硕士大论文/我的工作/py写弹道仿真代码 -最终/nn跑拟合')

# 加载数据

file_path = 'D:/硕士大论文/我的工作/py写弹道仿真代码 -最终/nn跑拟合/combined_output.csv'
data = pd.read_csv(file_path, header=1, encoding='gbk')

# 获取特征X
X = data.iloc[:, :5].values

# 获取标签y
y = data.iloc[:, 5:11].values

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_3round.pkl')

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
# 5 16 64/32 32 16 9
# 定义神经网络
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


# 初始化模型、损失函数和优化器
model = model_3round().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 训练模型
num_epochs = 20
batch_size = 32

for epoch in range(num_epochs):
    model.train()

    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i + batch_size]
        labels = y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('训练完成')

# 保存模型
torch.save(model.state_dict(), 'model_3round.pth')


# 测试模型
model.eval()  # 切换到评估模式
with torch.no_grad():  # 关闭梯度计算
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
