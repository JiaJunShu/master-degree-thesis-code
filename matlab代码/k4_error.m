% 读取 XLSX 文件
data = xlsread('k4error.xlsx');
x = data(:, 4);  % 第四列为横坐标
y1 = data(:, 1); % 第一列为纵坐标1
y2 = data(:, 5); % 第五列为纵坐标2
y3 = data(:, 6); % 第六列为纵坐标3

% 多项式拟合（例如 15 阶多项式）
p1 = polyfit(x, y1, 5);  % 第一列数据拟合
p2 = polyfit(x, y2, 5);  % 第五列数据拟合
p3 = polyfit(x, y3, 5);  % 第六列数据拟合

% 使用拟合参数计算预测值
y1_fit = polyval(p1, x);
y2_fit = polyval(p2, x);
y3_fit = polyval(p3, x);

% 创建图形窗口
figure;
set(gcf, 'Color', 'w');  % 设置背景颜色为白色

% 绘制第一列数据及拟合曲线
%plot(x, y1, 'ok', 'DisplayName', '第一列数据');  % 原始数据散点图
hold on;
plot(x, y1_fit, '-k', 'DisplayName', '总体');  % 拟合曲线

% 绘制第五列数据及拟合曲线
%plot(x, y2, 'or', 'DisplayName', '第五列数据');  % 原始数据散点图
plot(x, y2_fit, '-r', 'DisplayName', 'x方向');  % 拟合曲线

% 绘制第六列数据及拟合曲线
%plot(x, y3, 'og', 'DisplayName', '第六列数据');  % 原始数据散点图
plot(x, y3_fit, '-g', 'DisplayName', 'y方向');  % 拟合曲线

% 添加标签和图例
xlabel('k4');
ylabel('误差/m');
legend('show');  % 显示图例
%grid on;  % 显示网格
