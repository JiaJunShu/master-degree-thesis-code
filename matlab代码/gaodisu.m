% 读取文本文件
data = load('mtkl_method4_v280_6240,0.txt'); % 使用 load 读取数据

% 提取列
coords_x = data(:, 1); % 第一列坐标 (实际坐标)
coords_y = data(:, 2); % 第二列坐标 (实际坐标)
x_coords = data(:, 3);  % 第三列作为坐标系的X坐标
y_coords = data(:, 4);  % 第四列作为坐标系的Y坐标

% 定义参考点
reference_point = [6240,0];

% 计算每个坐标和参考点的距离
distances = sqrt((coords_x - reference_point(1)).^2 + (coords_y - reference_point(2)).^2);

% 准备绘图
figure;
set(gcf, 'Color', 'w');  % 设置背景颜色为白色
% 创建网格
[X, Y] = meshgrid(linspace(min(x_coords), max(x_coords), 100), ...
                 linspace(min(y_coords), max(y_coords), 100));

% 进行一次插值计算 Z 值
Z = griddata(x_coords, y_coords, distances, X, Y, 'linear'); % 使用线性插值

% 绘制曲面
surf(X, Y, Z, 'EdgeColor', 'none'); % 绘制曲面，并去掉边界线
colorbar; % 显示颜色条
xlabel('发射偏角/°');
ylabel('初始翼筒滚转角/°');
zlabel('误差/m');
%title('Surface Plot of Distances (Linear Interpolation)');
view(30, 30); % 设置视角
grid on;

% 添加等高线
hold on;  % 保持当前图形
contour3(X, Y, Z, 5, 'LineColor', 'k'); % 绘制等高线，20个等高线
hold off; % 释放图形
