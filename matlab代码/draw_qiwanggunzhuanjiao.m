% 设置文件名
filename = 'output.txt'; % 请根据实际文件名修改

% 方法1: 使用 readtable 读取文件
dataTable = readtable(filename);

% 提取前七列数据
firstColumn = dataTable{:, 1}; % 第一列
secondColumn = dataTable{:, 2}; % 第二列
thirdColumn = dataTable{:, 3}; % 第三列
seventhColumn = dataTable{:, 7}; % 第七列

% 检查并转换所有列为单元格的列
columnsToCheck = {firstColumn, secondColumn, thirdColumn, seventhColumn};
for i = 1:length(columnsToCheck)
    if iscell(columnsToCheck{i})
        columnsToCheck{i} = str2double(columnsToCheck{i});
    else
        columnsToCheck{i} = double(columnsToCheck{i});
    end
end

% 分配已转换的列
[firstColumn, secondColumn, thirdColumn, seventhColumn] = deal(columnsToCheck{:});

% 删除任何 NaN 值,确保数据是干净的
validIndices = ~(isnan(firstColumn) | isnan(secondColumn) | isnan(thirdColumn) | isnan(seventhColumn));
firstColumn = firstColumn(validIndices);
secondColumn = secondColumn(validIndices);
thirdColumn = thirdColumn(validIndices);
seventhColumn = seventhColumn(validIndices);

% 从第 15062 行开始提取数据
startIndex = 15062;
firstColumn = firstColumn(startIndex:end);
seventhColumn = seventhColumn(startIndex:end);

% 绘制第一列数据
figure; % 创建一个新图形窗口
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
plot(seventhColumn, firstColumn, 'r-'); % 绘制第一列数据,红色实线
xlabel('时间/t'); % 为 x 轴添加标签
ylabel('期望翼筒滚转角/°'); % 为 y 轴添加标签

% 注释掉或删除下面一行以不显示图例
% legend('show'); % 不调用这行，以不显示图例
