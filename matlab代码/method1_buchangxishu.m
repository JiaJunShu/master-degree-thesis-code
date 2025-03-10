% 定义横坐标和纵坐标
x = [-174,-165,-153,-133,-120,-109,-100,-90,-86,-70,-50,-30,-10,20,30,60,80,100,124,140,168,170];
y = [-163,-136,-132,-110,-95,-85,-82,-60,-50,-25,-17,-5,0,1.3,7,19,30,50,88,110,146,158];

% 计算比值
ratios = y ./ x;

% 创建图形
figure;
set(gcf, 'Color', 'w'); % 设置整个图形窗口背景为白色
% 使用插值生成平滑的曲线
xq = linspace(min(x), max(x),100); % 创建一个细分的x向量用于插值
ratios_smooth = interp1(x, ratios, xq, 'spline'); % 使用样条插值进行平滑

% 绘制原始数据点
%plot(x, ratios, 'o', 'DisplayName', 'Data Points');
hold on;

% 绘制平滑曲线
plot(xq, ratios_smooth, 'r-', 'DisplayName', 'Smooth Curve');

% 添加标题和坐标轴标签
%title('Smooth Connection of Ratios');
xlabel('补偿前的角度');
ylabel('K');

% 添加图例
legend show;

% 显示网格
%grid on;
hold off;
