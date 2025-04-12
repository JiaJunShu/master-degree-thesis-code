% 设置文件名
filename = 'output.txt'; % 请根据实际文件名修改

% 方法1: 使用 readtable 读取文件
dataTable = readtable(filename);

% 提取第七列数据
seventhColumn = dataTable{:, 7}; % 第七列
fourteenthColumn = dataTable{:, 14}; % 第十四列
thirteenthColumn = dataTable{:, 13}; % 第十三列

% 检查并转换所有列为单元格的列
if iscell(seventhColumn)
    seventhColumn = str2double(seventhColumn);
else
    seventhColumn = double(seventhColumn);
end

if iscell(fourteenthColumn)
    fourteenthColumn = str2double(fourteenthColumn);
else
    fourteenthColumn = double(fourteenthColumn);
end

if iscell(thirteenthColumn)
    thirteenthColumn = str2double(thirteenthColumn);
else
    thirteenthColumn = double(thirteenthColumn);
end

% 删除任何 NaN 值，确保数据是干净的
validIndices = ~(isnan(seventhColumn) | isnan(fourteenthColumn) | isnan(thirteenthColumn));
seventhColumn = seventhColumn(validIndices);
fourteenthColumn = fourteenthColumn(validIndices);
thirteenthColumn = thirteenthColumn(validIndices);

% 绘制第四列数据图
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(seventhColumn, fourteenthColumn, 'k-', 'DisplayName', '第十四列数据'); % 使用黑色实线绘制
xlabel('t(s)'); % 为 x 轴添加标签
ylabel('航迹倾斜角(°)'); % 为 y 轴添加标签
%title('第七列为横坐标 第十四列数据图'); % 添加标题
%grid on; % 添加网格
%legend('show'); % 可以选择不显示图例

% 绘制第十三列数据图
figure; % 创建另一个新图形窗口
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(seventhColumn, thirteenthColumn, 'k-', 'DisplayName', '第十三列数据'); % 使用黑色实线绘制
xlabel('t(s)'); % 为 x 轴添加标签
ylabel('航迹方位角(°)'); % 为 y 轴添加标签
%title('第七列为横坐标 第十三列数据图'); % 添加标题
%grid on; % 添加网格
%legend('show'); % 可以选择不显示图例
