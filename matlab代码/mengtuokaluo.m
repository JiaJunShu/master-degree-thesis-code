% 读取数据以分隔制表符和空格
data = readtable('mtkl_method1_wind_zong_v220_4320,0.txt', 'Delimiter', ' ', 'ReadVariableNames', false);
data2 = readtable('mtkl_method2_wind_zong_v220_4320,0.txt', 'Delimiter', ' ', 'ReadVariableNames', false);

% 提取每列数据
y = data{:, 1}; % 第一组的纵坐标
x = data{:, 2}; % 第一组的横坐标
y2 = data2{:, 1}; % 第二组的纵坐标
x2 = data2{:, 2}; % 第二组的横坐标

% 绘制散点图
scatter(x, y, 'o', 'filled'); % 第一组数据用圆点表示
hold on; % 保持当前图形
scatter(x2, y2, 'x'); % 第二组数据用三角形表示



% 高亮 (4350, 4300) 用红色标出
highlight_x = 0; % 高亮的 x 坐标
highlight_y = 4320; % 高亮的 y 坐标
scatter(highlight_x, highlight_y, 100, 'red', 'filled'); % 红点更大一些

% 添加标注
%text(highlight_x, highlight_y, '  (6275, 0)', 'Color', 'red', 'FontSize', 10); 

set(gcf, 'Color', 'w');  % 设置背景颜色为白色


% 以 (4300, 60) 为圆心画圆
circle_highlight_x = 0; % 圆心 x 坐标
circle_highlight_y = 4320; % 圆心 y 坐标
radius = 10; % 半径
theta = linspace(0, 2*pi, 100); % 圆周角度
circle_x = circle_highlight_x + radius * cos(theta); % 圆的 x 坐标
circle_y = circle_highlight_y + radius * sin(theta); % 圆的 y 坐标
plot(circle_x, circle_y, 'r-', 'LineWidth', 1.5); % 绘制圆，红色线条

% 添加标题和坐标轴标签
xlabel('Y/m');
ylabel('X/m');


% 添加图例
legend('方法一', '方法二','目标点', 'Location', 'northeast'); 

% 设置轴范围
xlim([-12, 12]); % 新的 x 坐标范围
ylim([4308,4332]); % 新的 y 坐标范围

hold off; % 结束当前图形的保持
