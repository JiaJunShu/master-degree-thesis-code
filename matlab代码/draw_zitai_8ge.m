% 设置文件名
filename = 'output.txt'; % 请根据实际文件名修改

% 使用 readtable 读取文件
dataTable = readtable(filename);

% 提取各列数据
seventhColumn = dataTable{:, 7}; % 第七列
firstColumn = dataTable{:, 1}; % 第一列
secondColumn = dataTable{:, 2}; % 第二列
thirdColumn = dataTable{:, 3}; % 第三列
eighthColumn = dataTable{:, 8}; % 第八列
ninthColumn = dataTable{:, 9}; % 第九列
tenthColumn = dataTable{:, 10}; % 第十列
eleventhColumn = dataTable{:, 11}; % 第十一列
twelfthColumn = dataTable{:, 12}; % 第十二列

% 检查并转换列为数值格式
columnsToCheck = {seventhColumn, firstColumn, secondColumn, thirdColumn, eighthColumn, ninthColumn, tenthColumn, eleventhColumn, twelfthColumn};
for i = 1:length(columnsToCheck)
    if iscell(columnsToCheck{i})
        columnsToCheck{i} = str2double(columnsToCheck{i});
    else
        columnsToCheck{i} = double(columnsToCheck{i});
    end
end

% 分配已转换的列
[seventhColumn, firstColumn, secondColumn, thirdColumn, eighthColumn, ninthColumn, tenthColumn, eleventhColumn, twelfthColumn] = deal(columnsToCheck{:});

% 删除任何 NaN 值，确保数据是干净的
validIndices = ~any(isnan([seventhColumn, firstColumn, secondColumn, thirdColumn, eighthColumn, ninthColumn, tenthColumn, eleventhColumn, twelfthColumn]), 2);
seventhColumn = seventhColumn(validIndices);
firstColumn = firstColumn(validIndices);
secondColumn = secondColumn(validIndices);
thirdColumn = thirdColumn(validIndices);
eighthColumn = eighthColumn(validIndices);
ninthColumn = ninthColumn(validIndices);
tenthColumn = tenthColumn(validIndices);
eleventhColumn = eleventhColumn(validIndices);
twelfthColumn = twelfthColumn(validIndices);

% 绘制图形
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(seventhColumn, firstColumn, 'k-', 'DisplayName', '第一列'); % 第一列数据
xlabel('t(s)'); % x 轴标签
ylabel('翼筒滚转角/°'); % y 轴标签
%grid on; % 添加网格

figure; % 创建另一个图形窗口
set(gcf, 'Color', 'w'); % 设置背景为白色
plot(seventhColumn, secondColumn, 'k-', 'DisplayName', '第二列'); % 第二列数据
xlabel('t(s)'); % x 轴标签
ylabel('弹体滚转角/°'); % y 轴标签
%grid on; % 添加网格

figure; % 创建另一个图形窗口
set(gcf, 'Color', 'w'); % 设置背景为白色
plot(seventhColumn, thirdColumn, 'k-', 'DisplayName', '第三列'); % 第三列数据
xlabel('t(s)'); % x 轴标签
ylabel('俯仰角/°'); % y 轴标签
%grid on; % 添加网格

% 绘制第八列图形
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(seventhColumn, eighthColumn, 'k-', 'DisplayName', '第八列'); % 第八列数据
xlabel('t(s)'); % x 轴标签
ylabel('偏航角/°'); % y 轴标签
grid off; % 关闭网格

% 绘制第九列图形
figure; % 创建另一个新图形窗口
set(gcf, 'Color', 'w'); % 设置背景为白色
plot(seventhColumn, ninthColumn, 'k-', 'DisplayName', '第九列'); % 第九列数据
xlabel('t(s)'); % x 轴标签
ylabel('第九列数据'); % y 轴标签
grid off; % 关闭网格

% 绘制第十列图形
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置背景为白色
plot(seventhColumn, tenthColumn, 'k-', 'DisplayName', '第十列'); % 第十列数据
xlabel('t(s)'); % x 轴标签
ylabel('第十列数据'); % y 轴标签
grid off; % 关闭网格

% 绘制第十一列图形
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置背景为白色
plot(seventhColumn, eleventhColumn, 'k-', 'DisplayName', '第十一列'); % 第十一列数据
xlabel('t(s)'); % x 轴标签
ylabel('俯仰角速度(°/s)'); % y 轴标签
grid off; % 关闭网格

% 绘制第十二列图形
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置背景为白色
plot(seventhColumn, twelfthColumn, 'k-', 'DisplayName', '第十二列'); % 第十二列数据
xlabel('t(s)'); % x 轴标签
ylabel('偏航角速度(°/s)'); % y 轴标签
grid off; % 关闭网格
