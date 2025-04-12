% 读取数据以分隔制表符和空格
data = readtable('round2_luodianfanwei_v220.txt', 'Delimiter', {' ', '\t'}, 'MultipleDelimsAsOne', true, 'ReadVariableNames', false);


% 提取每列数据
y = data{:, 1}; % 第一列作为纵坐标
x = data{:, 2}; % 第二列作为横坐标



% 创建散点图
figure;

highlight_x=0;
highlight_y=4320;
scatter(highlight_x, highlight_y, 100, 'red', 'filled');
text(highlight_x, highlight_y, '  (4320,0)', 'Color', 'red', 'FontSize', 10); 
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
%scatter(x, y, 'filled'); % 绘制散点图，'filled'表示点填充
hold on; % 保持当前图形
plot(x, y, 'k-'); % 绘制连线
plot([x(1), x(end)], [y(1), y(end)],  'k-'); % 连接最后一个点和第一个点
xlabel('侧偏y/m');
ylabel('射程x/m');
%title('落点范围');
%grid on; % 可以显示网格
text(0, 4445, '0°', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(99, 4312, '90°', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(-109, 4313, '-90°', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(-2, 4200, '180°', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');



% 在图上添加罗马数字"Ⅰ"
text(80, 4400, 'Ⅰ', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

% 在图上添加罗马数字"Ⅰ"
text(80, 4250, 'Ⅱ', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

% 在图上添加罗马数字"Ⅰ"
text(-90, 4250, 'Ⅲ', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

% 在图上添加罗马数字"Ⅰ"
text(-80, 4400, 'Ⅳ', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');

hold off; % 释放当前图形