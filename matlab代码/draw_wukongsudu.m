% 设置文件名
filename = 'output.txt'; % 请根据实际文件名修改

% 方法1: 使用 readtable 读取文件
dataTable = readtable(filename);

% 提取第七列数据
seventhColumn = dataTable{:, 7}; % 第七列
fifteenthColumn = dataTable{:, 15}; % 第十五列（确保文件包含这一列）

% 检查并转换所有列为单元格的列
if iscell(seventhColumn)
    seventhColumn = str2double(seventhColumn);
else
    seventhColumn = double(seventhColumn);
end

if iscell(fifteenthColumn)
    fifteenthColumn = str2double(fifteenthColumn);
else
    fifteenthColumn = double(fifteenthColumn);
end

% 删除任何 NaN 值，确保数据是干净的
validIndices = ~(isnan(seventhColumn) | isnan(fifteenthColumn));
seventhColumn = seventhColumn(validIndices);
fifteenthColumn = fifteenthColumn(validIndices);

% 绘制图形
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(seventhColumn, fifteenthColumn, 'k-', 'DisplayName', '第十五列数据'); % 使用黑色实线绘制
xlabel('t(s)'); % 为 x 轴添加标签
ylabel('v(m/s)'); % 为 y 轴添加标签
%title('第七列为横坐标 第十五列数据图'); % 添加标题
%grid on; % 添加网格
%legend('show'); % 显示图例
