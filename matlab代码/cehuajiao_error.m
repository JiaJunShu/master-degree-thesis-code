% 读取第一个txt文件中的数据
data1 = importdata('mtkl_method1_cehuajiao_v220_4320,0.txt');

% 提取每一列
x_coords1 = data1(:, 1);  % 第一列数据
y_coords1 = data1(:, 2);  % 第二列数据
z_coords1 = data1(:, 3);  % 第三列数据

% 计算与点 (4320, 0) 的距离
reference_point = [4320, 0];  % 参考点 (4320, 0)
distances1 = sqrt((x_coords1 - reference_point(1)).^2 + (y_coords1 - reference_point(2)).^2);

% 对 z_coords1 从小到大排序，并同步调整 distances1
[z_coords_sorted1, sort_idx1] = sort(z_coords1);
distances_sorted1 = distances1(sort_idx1);

% 使用 polyfit 进行多项式拟合（例如二次拟合）
p1 = polyfit(z_coords_sorted1, distances_sorted1, 5);  % 7 表示七次拟合

% 生成拟合曲线的 x 值
z_fit1 = linspace(min(z_coords_sorted1), max(z_coords_sorted1), 100);

% 计算拟合曲线的 y 值
distances_fit1 = polyval(p1, z_fit1);

% 读取第二个txt文件中的数据
data2 = importdata('mtkl_method2_cehuajiao_v220_4320,0.txt');  % 替换为第二个文件的实际路径

% 提取每一列
x_coords2 = data2(:, 1);  % 第一列数据
y_coords2 = data2(:, 2);  % 第二列数据
z_coords2 = data2(:, 3);  % 第三列数据

% 计算与点 (4320, 0) 的距离
distances2 = sqrt((x_coords2 - reference_point(1)).^2 + (y_coords2 - reference_point(2)).^2);

% 对 z_coords2 从小到大排序，并同步调整 distances2
[z_coords_sorted2, sort_idx2] = sort(z_coords2);
distances_sorted2 = distances2(sort_idx2);

% 使用 polyfit 进行多项式拟合（例如二次拟合）
p2 = polyfit(z_coords_sorted2, distances_sorted2, 5);  % 7 表示七次拟合

% 生成拟合曲线的 x 值
z_fit2 = linspace(min(z_coords_sorted2), max(z_coords_sorted2), 100);

% 计算拟合曲线的 y 值
distances_fit2 = polyval(p2, z_fit2);


% 绘制拟合曲线
figure;
%plot(z_coords_sorted1, distances_sorted1, 'o', 'DisplayName', '文件1原始数据');  % 文件1原始数据点
hold on;
set(gcf, 'Color', 'w');  % 设置背景颜色为白色
plot(z_fit1, distances_fit1, '-', 'DisplayName', '方法一');  % 文件1拟合曲线
%plot(z_coords_sorted2, distances_sorted2, 's', 'DisplayName', '文件2原始数据');  % 文件2原始数据点
plot(z_fit2, distances_fit2, '-', 'DisplayName', '方法二');  % 文件2拟合曲线
xlabel('初始射角/°');
ylabel('误差/m');
%title('两个文件的拟合曲线');
legend;
%grid on;
hold off;
