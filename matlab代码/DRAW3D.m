% 读取文件并提取数据
data = load('output.txt');

% 提取 x, y 和 z 数据
x = data(:, 4);
y = data(:, 5);
z = data(:, 6);

% 绘制3D图形
figure;
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot3(x, y, z, '-o', 'LineWidth', 0.1);

% 标注起点
hold on;
plot3(x(1), y(1), z(1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(x(1), y(1), z(1), ' 起点', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% 标注终点
plot3(x(end), y(end), z(end), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
text(x(end), y(end), z(end), ' 终点', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% 显示网格
grid on;
xlabel('X/m');
ylabel('Y/m');
zlabel('Z/m');

grid off; % 关闭网格
hold off;
