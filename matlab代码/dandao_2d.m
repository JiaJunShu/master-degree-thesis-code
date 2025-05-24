% 假设文件名为 "output.txt"
filename = 'output.txt';

% 使用制表符作为分隔符读取文件
opts = detectImportOptions(filename, 'Delimiter', '\t');

% 确保文件有足够的列，并指定读取的列，第四（Var4）和第五（Var5）列
opts.SelectedVariableNames = opts.VariableNames([4, 6]);

% 将数据读入表格
data = readtable(filename, opts);

% 将第四列和第五列转换为数组
x = table2array(data(:, 1)); % 第四列（横坐标）
y = table2array(data(:, 2)); % 第五列（纵坐标）

% 绘制坐标图
figure;
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(x, y, '-', 'LineWidth', 1, 'Color', 'k'); % 绘制连接线
hold on; % 保持当前图形

% 绘制起点和终点
plot(x(1), y(1), 'ro', 'MarkerSize', 8, 'LineWidth', 2); % 起点，红色圆圈
plot(x(end), y(end), 'bs', 'MarkerSize', 8, 'LineWidth', 2); % 终点，蓝色方块

% 直接在起点和终点位置添加文字注释
text(x(1), y(1), '起点', 'FontSize', 10, 'Color', 'black', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right'); % 起点标签
text(x(end), y(end), '终点', 'FontSize', 10, 'Color', 'black', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left'); % 终点标签

% 添加坐标轴标签和网格
xlabel('X/m');
ylabel('Z/m');

%grid on; % 显示网格
