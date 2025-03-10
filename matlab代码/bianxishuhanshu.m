% 定义x轴范围
x = -50:0.1:50;

% 定义函数
y1 = 1 ./ (1 + exp(-x));
y2 = 1 ./ (1 + exp(-x/10));
y3 = 1 ./ (1 + exp(-x/20));
y4 = 1 ./ (1 + exp(-x/50));

% 绘制图像
plot(x, y1, 'k-', 'LineWidth', 2); hold on;
plot(x, y2, 'k--', 'LineWidth', 2);
plot(x, y3, 'k-.', 'LineWidth', 2);
%plot(x, y4, 'k:', 'LineWidth', 2);
hold off;

set(gcf, 'Color', 'w'); % 设置背景颜色为白色

% 添加图例和题注
legend('k4=1', ...
       'k4=10', ...
       'k4=20', ...
       'k4=50');
xlabel('△x');
ylabel('sigmoid函数');
%title('函数 y = 1 / (1 + exp(x)) 及其变体比较图');
%grid on;
